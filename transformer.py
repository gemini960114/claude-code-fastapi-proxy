import json
from typing import Any, Dict, List, Optional, Tuple

def strip_anthropic_billing_header(text: str) -> str:
    key = "x-anthropic-billing-header:"
    if len(text) < len(key) or not text[:len(key)].lower() == key.lower():
        return text
    
    first_nl = text.find("\n")
    if first_nl == -1:
        return text
    
    after_first = text[first_nl + 1:]
    first_line = text[:first_nl]
    
    # Inline form
    if "cc_version=" in first_line or "cc_entrypoint=" in first_line:
        return after_first
    
    # Wrapped form
    next_nl = after_first.find("\n")
    if next_nl != -1:
        cont = after_first[:next_nl]
        if "cc_version=" in cont or "cc_entrypoint=" in cont or "cch=" in cont:
            return after_first[next_nl + 1:]
            
    return text

def strip_billing_header_in_content(content: Any) -> Any:
    if isinstance(content, str):
        return strip_anthropic_billing_header(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and "text" in block:
                block["text"] = strip_anthropic_billing_header(block["text"])
                break
    return content

def strip_billing_header_in_body(body: Dict[str, Any]) -> None:
    if "system" in body:
        body["system"] = strip_billing_header_in_content(body["system"])
    if "messages" in body and isinstance(body["messages"], list) and len(body["messages"]) > 0:
        first_msg = body["messages"][0]
        if isinstance(first_msg, dict) and "content" in first_msg:
            first_msg["content"] = strip_billing_header_in_content(first_msg["content"])

def extract_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content) if content is not None else ""

def convert_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "none":
            return "none"
        if choice_type == "tool" and tool_choice.get("name"):
            return {"type": "function", "function": {"name": tool_choice.get("name")}}
    return None

def validate_and_fix_tool_message_order(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fixed: List[Dict[str, Any]] = []
    pending_tool_call_ids: set = set()

    for msg in messages:
        role = msg.get("role")

        if role == "assistant":
            fixed.append(msg)
            pending_tool_call_ids.clear()
            for call in msg.get("tool_calls") or []:
                if isinstance(call, dict):
                    call_id = call.get("id")
                    if call_id:
                        pending_tool_call_ids.add(call_id)
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id and tool_call_id in pending_tool_call_ids:
                fixed.append(msg)
                pending_tool_call_ids.discard(tool_call_id)
            else:
                fixed.append({
                    "role": "user",
                    "content": "[Tool result without matching tool call]\n" + str(msg.get("content", ""))
                })
            continue

        fixed.append(msg)
        if role in ("user", "system"):
            pending_tool_call_ids.clear()

    return fixed

def anthropic_document_to_file_part(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source = block.get("source") or {}
    source_type = source.get("type")
    filename = block.get("title") or "document.pdf"

    if source_type == "base64":
        media_type = source.get("media_type", "application/pdf")
        data = source.get("data", "")
        if not data:
            return None
        return {
            "type": "file",
            "file": {
                "filename": filename,
                "file_data": f"data:{media_type};base64,{data}"
            }
        }
    if source_type == "url":
        url = source.get("url")
        if not url:
            return None
        return {"type": "file", "file": {"filename": filename, "file_data": url}}
    return None

def convert_media_block(block: Dict[str, Any], parts: List[Dict[str, Any]]) -> None:
    block_type = block.get("type", "")
    if block_type == "image":
        source = block.get("source") or {}
        source_type = source.get("type")
        if source_type == "base64":
            data = source.get("data")
            media_type = source.get("media_type") or block.get("media_type", "image/png")
            if data:
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{data}"
                    }
                })
        elif source_type == "url":
            url = source.get("url")
            if url:
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": url
                    }
                })
    elif block_type == "document":
        import os
        enable_doc = os.environ.get("ENABLE_DOCUMENT_PART", "true").lower() == "true"
        if enable_doc:
            file_part = anthropic_document_to_file_part(block)
            if file_part:
                parts.append(file_part)
        else:
            filename = block.get("title") or "document.pdf"
            parts.append({
                "type": "text",
                "text": f"[PDF document omitted: {filename}]"
            })
    elif "text" in block:
        parts.append({
            "type": "text",
            "text": block["text"]
        })

def text_parts_to_content(parts: List[Dict[str, Any]]) -> Any:
    if not parts:
        return ""
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0].get("text", "")
    return parts

def convert_messages(anth_messages: List[Dict[str, Any]], system_prompt: Optional[Any] = None) -> List[Dict[str, Any]]:
    openai_messages = []

    # System prompt translation
    if system_prompt:
        if isinstance(system_prompt, str):
            openai_messages.append({
                "role": "system",
                "content": system_prompt
            })
        elif isinstance(system_prompt, list):
            text = "\n".join([b.get("text", "") for b in system_prompt if isinstance(b, dict) and "text" in b])
            if text:
                openai_messages.append({
                    "role": "system",
                    "content": text
                })

    for msg in anth_messages:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            openai_role = {
                "user": "user",
                "assistant": "assistant",
                "system": "system",
                "developer": "developer"
            }.get(role)
            if openai_role:
                openai_messages.append({
                    "role": openai_role,
                    "content": content
                })
            continue

        if isinstance(content, list):
            if role == "assistant":
                text_parts = []
                tool_calls = []

                for block in content:
                    block_type = block.get("type", "")
                    if block_type == "text":
                        if "text" in block:
                            text_parts.append({
                                "type": "text",
                                "text": block["text"]
                            })
                    elif block_type == "tool_use":
                        id_ = block.get("id")
                        name = block.get("name")
                        input_val = block.get("input", {})
                        if id_ and name:
                            tool_calls.append({
                                "id": id_,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(input_val)
                                }
                            })
                    else:
                        convert_media_block(block, text_parts)

                msg_obj = {"role": "assistant"}
                if not tool_calls:
                    msg_obj["content"] = text_parts_to_content(text_parts)
                else:
                    msg_obj["content"] = text_parts_to_content(text_parts) if text_parts else None
                    msg_obj["tool_calls"] = tool_calls
                openai_messages.append(msg_obj)

            elif role == "user":
                text_parts = []
                tool_results = []

                for block in content:
                    block_type = block.get("type", "")
                    if block_type == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        result_content = extract_tool_result_content(block.get("content"))
                        tool_results.append((tool_use_id, result_content))
                    elif block_type == "text":
                        if "text" in block:
                            text_parts.append({
                                "type": "text",
                                "text": block["text"]
                            })
                    else:
                        convert_media_block(block, text_parts)

                # Tool results must come before user text
                for tool_call_id, result in tool_results:
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result
                    })

                if text_parts:
                    openai_messages.append({
                        "role": "user",
                        "content": text_parts_to_content(text_parts)
                    })
                    
            elif role in ("system", "developer"):
                text = "\n".join([b.get("text", "") for b in content if isinstance(b, dict) and "text" in b])
                openai_messages.append({
                    "role": role,
                    "content": text
                })

    return openai_messages

def transform_anthropic_to_openai(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "model" not in body or "messages" not in body:
        return None
        
    model = body["model"]
    messages = body["messages"]
    system = body.get("system")
    stream = body.get("stream", False)

    openai_messages = convert_messages(messages, system)
    openai_messages = validate_and_fix_tool_message_order(openai_messages)

    result = {
        "model": model,
        "messages": openai_messages,
        "stream": stream
    }

    # Tools conversion
    if "tools" in body and isinstance(body["tools"], list):
        openai_tools = []
        for tool in body["tools"]:
            name = tool.get("name")
            description = tool.get("description", "")
            input_schema = tool.get("input_schema", {})
            if name:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": input_schema
                    }
                })
        if openai_tools:
            result["tools"] = openai_tools
            tool_choice = convert_tool_choice(body.get("tool_choice"))
            if tool_choice is not None:
                result["tool_choice"] = tool_choice

    # Pass through standard parameters
    for key in ["temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty"]:
        if key in body:
            result[key] = body[key]

    if "stop_sequences" in body:
        result["stop"] = body["stop_sequences"]

    import os
    if stream and os.environ.get("STREAM_INCLUDE_USAGE", "true").lower() == "true":
        result["stream_options"] = {"include_usage": True}

    return result

def transform_openai_response_to_anthropic(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})

    content_blocks = []

    content = message.get("content")
    if content and isinstance(content, str):
        content_blocks.append({
            "type": "text",
            "text": content
        })

    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        fn = tc.get("function", {})
        name = fn.get("name", "")
        arguments = fn.get("arguments", "{}")
        try:
            input_val = json.loads(arguments)
        except Exception:
            input_val = {}

        content_blocks.append({
            "type": "tool_use",
            "id": tc_id,
            "name": name,
            "input": input_val
        })

    finish_reason = choice.get("finish_reason", "end_turn")
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use"
    }.get(finish_reason, finish_reason)

    return {
        "id": response.get("id", ""),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": response.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": response.get("usage", {
            "input_tokens": 0,
            "output_tokens": 0
        })
    }

def normalize_openai_tools_in_chat_body(body: Dict[str, Any]) -> None:
    """Normalize tools schema if description lacks type, etc."""
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "function":
                function = tool.get("function")
                if isinstance(function, dict):
                    parameters = function.get("parameters")
                    if isinstance(parameters, dict) and "type" not in parameters:
                        parameters["type"] = "object"

def sse_event(data: Dict[str, Any]) -> str:
    event_type = data.get("type", "message")
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

class StreamState:
    def __init__(self):
        self.is_first = True
        self.accumulated_content = ""
        self.text_block_index = None
        self.tool_blocks = {} # tc_index -> block_index
        self.next_block_index = 0
        self.finished = False

def transform_openai_chunk_to_anthropic_events(
    line: str, 
    state: StreamState
) -> List[str]:
    """
    Parses a single 'data: {...}' line from an OpenAI stream chunk, 
    updating the StreamState and yielding translated Anthropic SSE events.
    """
    if state.finished:
        return []
        
    if not line.startswith("data:"):
        return []
        
    data_str = line[5:].strip()
    if not data_str:
        return []
        
    events = []
    
    if data_str == "[DONE]":
        state.finished = True
        # Close open text block
        if state.text_block_index is not None:
            stop = {"type": "content_block_stop", "index": state.text_block_index}
            events.append(sse_event(stop))
            state.text_block_index = None
            
        # Close open tool blocks
        for idx in sorted(state.tool_blocks.values()):
            stop = {"type": "content_block_stop", "index": idx}
            events.append(sse_event(stop))
        state.tool_blocks.clear()
        
        stop_reason = "end_turn" if not state.tool_blocks else "tool_use"
        output_tokens = len(state.accumulated_content.split())
        
        delta_event = {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None
            },
            "usage": { "output_tokens": output_tokens }
        }
        events.append(sse_event(delta_event))
        
        message_stop = {"type": "message_stop"}
        events.append(sse_event(message_stop))
        return events

    try:
        json_chunk = json.loads(data_str)
    except Exception:
        return []

    choices = json_chunk.get("choices", [])
    if not choices:
        return []
    choice = choices[0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")
    has_finish = finish_reason is not None

    if state.is_first:
        role = delta.get("role", "assistant")
        message_id = json_chunk.get("id", "")
        model = json_chunk.get("model", "")
        
        start_event = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": role,
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": { "input_tokens": 0, "output_tokens": 0 }
            }
        }
        events.append(sse_event(start_event))
        state.is_first = False

    # Handle text content
    content = delta.get("content")
    if content and isinstance(content, str):
        if state.text_block_index is None:
            idx = state.next_block_index
            state.next_block_index += 1
            state.text_block_index = idx
            
            block_start = {
                "type": "content_block_start",
                "index": idx,
                "content_block": { "type": "text", "text": "" }
            }
            events.append(sse_event(block_start))
            
        state.accumulated_content += content
        delta_event = {
            "type": "content_block_delta",
            "index": state.text_block_index,
            "delta": { "type": "text_delta", "text": content }
        }
        events.append(sse_event(delta_event))

    # Handle tool calls
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        # Close active text block before tool blocks
        if state.text_block_index is not None:
            stop = {"type": "content_block_stop", "index": state.text_block_index}
            events.append(sse_event(stop))
            state.text_block_index = None

        for tc in tool_calls:
            tc_index = tc.get("index", 0)
            
            # New tool call
            if "id" in tc:
                id_ = tc.get("id", "")
                fn = tc.get("function", {})
                name = fn.get("name", "")
                
                idx = state.next_block_index
                state.next_block_index += 1
                state.tool_blocks[tc_index] = idx
                
                block_start = {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": id_,
                        "name": name,
                        "input": {}
                    }
                }
                events.append(sse_event(block_start))
                
            # Argument delta
            fn = tc.get("function", {})
            args = fn.get("arguments")
            if args and isinstance(args, str):
                if tc_index in state.tool_blocks:
                    idx = state.tool_blocks[tc_index]
                    delta_event = {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": args
                        }
                    }
                    events.append(sse_event(delta_event))

    if has_finish:
        state.finished = True
        # Close text block
        if state.text_block_index is not None:
            stop = {"type": "content_block_stop", "index": state.text_block_index}
            events.append(sse_event(stop))
            state.text_block_index = None
            
        # Close all tool blocks
        for idx in sorted(state.tool_blocks.values()):
            stop = {"type": "content_block_stop", "index": idx}
            events.append(sse_event(stop))
        state.tool_blocks.clear()

        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use"
        }.get(finish_reason, finish_reason)
        
        output_tokens = len(state.accumulated_content.split())
        
        delta_event = {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None
            },
            "usage": { "output_tokens": output_tokens }
        }
        events.append(sse_event(delta_event))
        
        message_stop = {"type": "message_stop"}
        events.append(sse_event(message_stop))

    return events
