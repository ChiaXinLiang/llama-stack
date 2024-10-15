import asyncio
import json
from typing import Any, AsyncGenerator, List, Optional

import httpx

class SimpleClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def chat_completion(
        self,
        model: str,
        messages: List[dict],
        stream: Optional[bool] = False,
    ) -> AsyncGenerator:
        request = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/inference/chat_completion",
                json=request,
                headers={"Content-Type": "application/json"},
                timeout=20,
            ) as response:
                if response.status_code != 200:
                    content = await response.aread()
                    print(f"Error: HTTP {response.status_code} {content.decode()}")
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data = line[len("data: ") :]
                        try:
                            yield json.loads(data)
                        except Exception as e:
                            print(f"Raw data: {data}")
                            print(f"Error with parsing or validation: {e}")

    async def text_completion(self, model: str, prompt: str, stream: Optional[bool] = False) -> AsyncGenerator:
        request = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/inference/text_completion",
                json=request,
                headers={"Content-Type": "application/json"},
                timeout=20,
            ) as response:
                if response.status_code != 200:
                    content = await response.aread()
                    print(f"Error: HTTP {response.status_code} {content.decode()}")
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data = line[len("data: ") :]
                        try:
                            yield json.loads(data)
                        except Exception as e:
                            print(f"Raw data: {data}")
                            print(f"Error with parsing or validation: {e}")

    async def embeddings(self, model: str, input: List[str]) -> dict:
        request = {
            "model": model,
            "input": input,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/inference/embeddings",
                json=request,
                headers={"Content-Type": "application/json"},
                timeout=20,
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: HTTP {response.status_code} {response.text}")
                return None

    async def memory_add(self, collection: str, documents: List[dict]) -> dict:
        request = {
            "collection": collection,
            "documents": documents,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/memory/add",
                json=request,
                headers={"Content-Type": "application/json"},
                timeout=20,
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: HTTP {response.status_code} {response.text}")
                return None

    async def memory_get(self, collection: str, ids: List[str]) -> dict:
        params = {
            "collection": collection,
            "ids": ids,
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory/get",
                params=params,
                timeout=20,
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: HTTP {response.status_code} {response.text}")
                return None

    async def memory_delete(self, collection: str, ids: List[str]) -> dict:
        params = {
            "collection": collection,
            "ids": ids,
        }
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.base_url}/memory/delete",
                params=params,
                timeout=20,
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: HTTP {response.status_code} {response.text}")
                return None

    async def memory_search(self, collection: str, query: str, k: int = 5) -> dict:
        params = {
            "collection": collection,
            "query": query,
            "k": k,
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory/search",
                params=params,
                timeout=20,
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: HTTP {response.status_code} {response.text}")
                return None

async def run_simple_client():
    client = SimpleClient("http://localhost:7777")

    # Chat Completion example
    print("Chat Completion:")
    messages = [{"role": "user", "content": "hello world, write me a 2 sentence poem about the moon"}]
    try:
        async for response in client.chat_completion(model="Llama3.2-3B", messages=messages, stream=True):
            print(response)
    except Exception as e:
        print(f"Error in chat_completion: {e}")

    # Text Completion example
    print("\nText Completion:")
    prompt = "Once upon a time, in a galaxy far, far away"
    try:
        async for response in client.text_completion(model="Llama3.2-3B", prompt=prompt, stream=True):
            print(response)
    except Exception as e:
        print(f"Error in text_completion: {e}")

    # Embeddings example
    print("\nEmbeddings:")
    input_texts = ["Hello, world!", "How are you?"]
    try:
        embeddings_result = await client.embeddings(model="Llama3.2-3B", input=input_texts)
        print(embeddings_result)
    except Exception as e:
        print(f"Error in embeddings: {e}")

    # Memory Add example
    print("\nMemory Add:")
    documents = [
        {"id": "doc1", "text": "This is the first document", "metadata": {"source": "user"}},
        {"id": "doc2", "text": "This is the second document", "metadata": {"source": "system"}},
    ]
    try:
        add_result = await client.memory_add(collection="my_collection", documents=documents)
        print(add_result)
    except Exception as e:
        print(f"Error in memory_add: {e}")

    # Memory Get example
    print("\nMemory Get:")
    try:
        get_result = await client.memory_get(collection="my_collection", ids=["doc1", "doc2"])
        print(get_result)
    except Exception as e:
        print(f"Error in memory_get: {e}")

    # Memory Search example
    print("\nMemory Search:")
    try:
        search_result = await client.memory_search(collection="my_collection", query="first document", k=1)
        print(search_result)
    except Exception as e:
        print(f"Error in memory_search: {e}")

    # Memory Delete example
    print("\nMemory Delete:")
    try:
        delete_result = await client.memory_delete(collection="my_collection", ids=["doc1"])
        print(delete_result)
    except Exception as e:
        print(f"Error in memory_delete: {e}")

if __name__ == "__main__":
    asyncio.run(run_simple_client())
