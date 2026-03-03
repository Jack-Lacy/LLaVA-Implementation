import requests, json
resp = requests.post("http://localhost:11434/v1/chat/completions", json={
    "model": "llava:13b",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 20
}, timeout=60)
print(resp.status_code)
print(resp.text[:1000])
