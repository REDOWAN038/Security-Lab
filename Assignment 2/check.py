def pkcs7_unpad(data1, data2):
    pad_size = data1[-1]
    print("\nsz : ", pad_size)
    print(type(pad_size))
    byte = bytes.fromhex(data2)
    print("\nb : ", byte)
    if pad_size < 1 or pad_size > len(data1):
        raise ValueError("Invalid padding size")
    if data1[-pad_size:] != bytes([pad_size] * pad_size):
        raise ValueError("Invalid padding bytes")
    return data1[:-pad_size], byte[:-pad_size].hex()

# Example usage:
encrypted_data = 'BUETnightfallVsSUSTguessforce\x03\x03\x03'
encrypted_data_hex = '425545546e6967687466616c6c5673535553546775657373666f726365030303'

decrypted_data, decrypted_data_hex = pkcs7_unpad(encrypted_data.encode('utf-8'), encrypted_data_hex)
# decrypted_data_hex = pkcs7_unpad_hex()

print("Decrypted data without padding:", decrypted_data.decode('utf-8'))
print("Decrypted data without padding:", decrypted_data_hex)

print(type(encrypted_data))
