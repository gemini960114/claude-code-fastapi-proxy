import unittest
import json
from transformer import (
    strip_anthropic_billing_header,
    strip_billing_header_in_body,
    convert_messages,
    transform_anthropic_to_openai,
    transform_openai_response_to_anthropic,
    transform_openai_chunk_to_anthropic_events,
    StreamState
)

class TestProxyTranslation(unittest.TestCase):
    
    def test_strip_billing_header_inline(self):
        text = "x-anthropic-billing-header: cc_version=0.1;\nactual content"
        self.assertEqual(strip_anthropic_billing_header(text), "actual content")
        
    def test_strip_billing_header_wrapped(self):
        text = "x-anthropic-billing-header:\n   cc_version=0.1;\nactual content"
        self.assertEqual(strip_anthropic_billing_header(text), "actual content")

    def test_strip_billing_header_not_present(self):
        text = "normal user query"
        self.assertEqual(strip_anthropic_billing_header(text), "normal user query")

    def test_convert_messages_simple(self):
        anth_msgs = [{"role": "user", "content": "Hello"}]
        converted = convert_messages(anth_msgs)
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["role"], "user")
        self.assertEqual(converted[0]["content"], "Hello")

    def test_convert_messages_with_system(self):
        anth_msgs = [{"role": "user", "content": "Hello"}]
        converted = convert_messages(anth_msgs, system_prompt="System instructions")
        self.assertEqual(len(converted), 2)
        self.assertEqual(converted[0]["role"], "system")
        self.assertEqual(converted[0]["content"], "System instructions")
        self.assertEqual(converted[1]["role"], "user")

    def test_convert_messages_assistant_tool_use(self):
        anth_msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Thinking..."},
                    {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}}
                ]
            }
        ]
        converted = convert_messages(anth_msgs)
        self.assertEqual(len(converted), 1)
        msg = converted[0]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "Thinking...")
        self.assertEqual(len(msg["tool_calls"]), 1)
        self.assertEqual(msg["tool_calls"][0]["id"], "call_1")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(msg["tool_calls"][0]["function"]["arguments"]), {"city": "Paris"})

    def test_transform_openai_response_to_anthropic(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello world!",
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            }
        }
        anth = transform_openai_response_to_anthropic(openai_resp)
        self.assertEqual(anth["id"], "chatcmpl-123")
        self.assertEqual(anth["type"], "message")
        self.assertEqual(anth["role"], "assistant")
        self.assertEqual(len(anth["content"]), 1)
        self.assertEqual(anth["content"][0]["type"], "text")
        self.assertEqual(anth["content"][0]["text"], "Hello world!")
        self.assertEqual(anth["stop_reason"], "end_turn")

    def test_stream_translation(self):
        state = StreamState()
        chunk_lines = [
            'data: {"id":"chat-123","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"id":"chat-123","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
            'data: {"id":"chat-123","model":"gpt-4","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
            'data: {"id":"chat-123","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            'data: [DONE]'
        ]
        
        events = []
        for line in chunk_lines:
            events.extend(transform_openai_chunk_to_anthropic_events(line, state))
            
        # Verify the sequence of generated events
        # We expect: message_start, content_block_start, content_block_delta (Hello), content_block_delta ( world), content_block_stop, message_delta, message_stop
        event_types = []
        for ev in events:
            lines = ev.strip().split("\n")
            for l in lines:
                if l.startswith("event:"):
                    event_types.append(l[6:].strip())
        
        expected_types = [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]
        self.assertEqual(event_types, expected_types)

if __name__ == "__main__":
    unittest.main()
