import asyncio
import websockets
import json
import aes
import rsa
    


async def send_data():
    uri = 'ws://localhost:8765'
    async with websockets.connect(uri) as websocket:
        key = "Thats my Kung Fu"
        plaintext = "Two One Nine Two"

        public_key, private_key = rsa.generate_key_pairs(16)
        keycipher = rsa.encrypt(key, public_key)

        key = aes.process_aes_key(key)
        padded_message = aes.pkcs7_pad(plaintext.encode('utf-8'), 16)
        aes.key_expansion(key)

        ciphertext, _ = aes.aes_encryption(aes.convert_bytes_to_hex(padded_message))
        keycipher = json.dumps(keycipher).encode()
        private_key = json.dumps(private_key)

        with open("Don't Open This/PRK.txt", "w") as f:
            f.write(private_key)


        await websocket.send(ciphertext)
        await websocket.send(keycipher)


if __name__=="__main__":
    asyncio.run(send_data())