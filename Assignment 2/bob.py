import asyncio
import websockets
import json
import aes
import rsa

async def receive_data(websocket):
    ciphertext = await websocket.recv()
    keycipher = await websocket.recv()

    with open("Don't Open This/PRK.txt", "r") as f:
        private_key = f.read()


    keycipher = json.loads(keycipher.decode())
    private_key = json.loads(private_key)

    key = rsa.decrypt(keycipher, private_key)
    key = aes.process_aes_key(key)
    aes.key_expansion(key)
    plaintext, _ = aes.aes_decryption(ciphertext)
    
    with open("Don't Open This/DPT.txt", "w") as f:
        f.write(plaintext)

    
async def main():
    async with websockets.serve(receive_data, "localhost", "8765"):
        await asyncio.Future()

if __name__=="__main__":
    asyncio.run(main())