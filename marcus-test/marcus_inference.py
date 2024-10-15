import asyncio
from marcus_client import SimpleClient

async def run_chat_completion():
    client = SimpleClient("http://localhost:7777")
    print("Chat Completion:")
    messages = [{"role": "user", "content": "Explain the concept of recursion in programming"}]
    try:
        async for chunk in client.chat_completion(model="Llama3.2-3B", messages=messages, stream=True):
            print(chunk, end='', flush=True)
        print()  # New line after streaming is complete
        
        # Get final result
        final_result = await client.chat_completion(model="Llama3.2-3B", messages=messages, stream=False)
        print("Final result:", final_result)
    except Exception as e:
        print(f"Error in chat_completion: {e}")

async def run_text_completion():
    client = SimpleClient("http://localhost:7777")
    print("\nText Completion:")
    prompt = "The benefits of artificial intelligence in healthcare include"
    try:
        async for chunk in client.text_completion(model="Llama3.2-3B", prompt=prompt, stream=True):
            print(chunk, end='', flush=True)
        print()  # New line after streaming is complete
        
        # Get final result
        final_result = await client.text_completion(model="Llama3.2-3B", prompt=prompt, stream=False)
        print("Final result:", final_result)
    except Exception as e:
        print(f"Error in text_completion: {e}")

async def run_embeddings():
    client = SimpleClient("http://localhost:7777")
    print("\nEmbeddings:")
    input_texts = ["Artificial Intelligence", "Machine Learning", "Deep Learning"]
    try:
        embeddings_result = await client.embeddings(model="Llama3.2-3B", contents=input_texts)
        print("Embeddings result:", embeddings_result)
    except Exception as e:
        print(f"Error in embeddings: {e}")

async def main():
    await run_chat_completion()
    await run_text_completion()
    await run_embeddings()

if __name__ == "__main__":
    asyncio.run(main())
