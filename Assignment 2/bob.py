import asyncio
import websockets

async def receive_data(websocket):
    ciphertext = await websocket.recv()
    hex_ciphertext = await websocket.recv()
    print("bob received ciphertext : ", ciphertext)
    print("bob received hex_ciphertext : ", hex_ciphertext)

    cipher = ciphertext + hex_ciphertext
    await websocket.send(cipher)
    print("bob sent cipher : ", cipher)
    
async def main():
    async with websockets.serve(receive_data, "localhost", "8765"):
        await asyncio.Future()

if __name__=="__main__":
    asyncio.run(main())