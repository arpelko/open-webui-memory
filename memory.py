"""
title: Memory
description: A tool to manage user memories in Open WebUI.
author: arpelko, ronilaukkarinen
version: 0.1.3
"""

import json
import logging
import ast
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import Request
from apps.webui.models.users import Users
from apps.webui.routers.memories import (
    AddMemoryForm,
    add_memory,
    get_memories,
    delete_memory_by_id,
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Action:
    class Valves(BaseModel):
        consolidation_model: str = Field(
            default="gpt-4o-mini",
            description="The model to use for consolidating memories.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Process the message before it reaches the model.
        Detects if the model wants to add a memory and handles it.
        """
        messages = body.get("messages", [])
        if not messages or not __user__:
            return body

        last_message = messages[-1]
        content = last_message.get("content", "")

        # Check if the model output contains a memory tag
        # Format: [MEMORY: some information to remember]
        memory_tag = "[MEMORY:"
        if memory_tag in content:
            start_index = content.find(memory_tag) + len(memory_tag)
            end_index = content.find("]", start_index)

            if end_index != -1:
                memory_to_add = content[start_index:end_index].strip()
                user_id = __user__.get("id")

                if user_id and memory_to_add:
                    print(f"Auto Memory: Attempting to store memory: '{memory_to_add}'")
                    
                    # Process and consolidate memory
                    status = await self.consolidate_memory(
                        memory_to_add, __user__, body.get("webui_app")
                    )
                    
                    if status is True:
                        print(f"Auto Memory: Successfully processed memory for user {user_id}")
                    else:
                        print(f"Auto Memory: Error processing memory: {status}")

                # Remove the memory tag from the content so the user doesn't see it
                new_content = content[: content.find(memory_tag)].strip()
                last_message["content"] = new_content

        return body

    async def consolidate_memory(self, memory: str, user: dict, webui_app) -> bool:
        """
        Consolidates the new memory with existing ones using an LLM.
        """
        try:
            # 1. Get existing memories
            existing_memories_data = get_memories(user.get("id"))
            existing_memories = [m.content for m in existing_memories_data]

            # FIX: Limit to the last 20 memories to prevent context window overflow
            # and reduce processing time.
            if len(existing_memories) > 20:
                print(f"Auto Memory: Limiting consolidation to the last 20 memories (total: {len(existing_memories)})")
                existing_memories = existing_memories[-20:]

            # 2. Use LLM to decide what to do
            system_prompt = (
                "You are a memory consolidation assistant. Your goal is to keep a user's long-term memory clean and concise. "
                "You are given a list of existing memories and a new memory to add. "
                "You must return a Python-style list of strings representing the updated memories. "
                "Rules:\n"
                "1. If the new memory provides updated information (e.g., 'User lives in Helsinki' followed by 'User moved to Tampere'), replace the old one.\n"
                "2. If the new memory is a duplicate or redundant, ignore it.\n"
                "3. If the new memory is new information, add it to the list.\n"
                "4. Combine related memories if it makes sense to save space.\n"
                "5. OUTPUT ONLY THE PYTHON LIST. No explanations. Example format: ['memory 1', 'memory 2']"
            )

            user_prompt = f"Existing memories: {existing_memories}\nNew memory: {memory}"

            # Call the internal LLM via Open WebUI's generate function
            # Note: We assume the environment provides access to a completion function
            # For this implementation, we use the model defined in valves.
            
            # This is a simplified call structure for Open WebUI functions
            from apps.olama.main import generate_chat_completion

            response = await generate_chat_completion(
                Request(scope={"type": "http", "app": webui_app}),
                {
                    "model": self.valves.consolidation_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
                user=user,
            )

            consolidated_memories_raw = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # Clean up potential markdown formatting if the model included it
            if consolidated_memories_raw.startswith("```python"):
                consolidated_memories_raw = consolidated_memories_raw[9:-3].strip()
            elif consolidated_memories_raw.startswith("```"):
                consolidated_memories_raw = consolidated_memories_raw[3:-3].strip()

            try:
                # Parse the string representation of the list
                consolidated_list = ast.literal_eval(consolidated_memories_raw)
                
                if not isinstance(consolidated_list, list):
                    raise ValueError("Output is not a list")

                # 3. Update database: Delete old (subset) and add new consolidated ones
                # We only delete the ones we actually used for consolidation
                for m_obj in existing_memories_data:
                    if m_obj.content in existing_memories:
                        delete_memory_by_id(m_obj.id)

                # Add consolidated ones
                for m_content in consolidated_list:
                    await add_memory(
                        request=Request(scope={"type": "http", "app": webui_app}),
                        form_data=AddMemoryForm(content=m_content),
                        user=user,
                    )
                
                return True

            except (ValueError, SyntaxError) as e:
                # FIX: Handle parsing errors with a robust fallback
                print(f"Auto Memory: Consolidation parsing failed. Response was: {repr(consolidated_memories_raw[:100])}...")
                print(f"Auto Memory: Falling back to storing memory without consolidation: '{memory}'")
                
                await add_memory(
                    request=Request(scope={"type": "http", "app": webui_app}),
                    form_data=AddMemoryForm(content=memory),
                    user=user,
                )
                return True

        except Exception as e:
            logger.error(f"Memory consolidation error: {e}")
            return f"Unable to consolidate memories: {str(e)}"
