import time
from BitVector_Helper import *


BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET_FORMATTING = "\033[0m"

round_keys = []
round_constants = [
    ['01', '00', '00', '00'],
    ['02', '00', '00', '00'],
    ['04', '00', '00', '00'],
    ['08', '00', '00', '00'],
    ['10', '00', '00', '00'],
    ['20', '00', '00', '00'],
    ['40', '00', '00', '00'],
    ['80', '00', '00', '00'],
    ['1b', '00', '00', '00'],
    ['36', '00', '00', '00'],
]

def process_aes_key(key):
    sz = len(key)

    if sz==16:
        return key
    elif sz>16:
        return key[:16]
    else:
        return key.ljust(16, 'X')

def pkcs7_pad(data, block_size):
    if(len(data)%block_size==0):
        return data
    pad_size = block_size - len(data) % block_size
    padding = bytes([pad_size] * pad_size)
    return data + padding

def pkcs7_unpad(data1, data2):
    pad_size = int(data1[-1])

    if(not(pad_size>=1 and pad_size<=16)):
        return data1, data2
        
    # byte = bytes.fromhex(data2)
    # if pad_size < 1 or pad_size > len(data1):
    #     raise ValueError("Invalid padding size")
    # if data1[-(pad_size*4):] != bytes([pad_size] * pad_size):
    #     raise ValueError("Invalid padding bytes")
    return data1[:-(pad_size)], data2[:-(pad_size*2)]

def convert_bytes_to_hex(bytes_object):
    return bytes_object.hex()

def convert_string_to_hex(str):
    hex_representation = ""
    for char in str:
        hex_char = hex(ord(char))[2:]
        if len(hex_char) == 1:
            hex_char = '0' + hex_char
        hex_representation += hex_char
    return hex_representation
    # bytes_object = str.encode('utf-8')
    # return convert_bytes_to_hex(bytes_object)

def convert_hex_to_string(hex_string):
    decimal_value = int(hex_string, 16)
    unicode_character = chr(decimal_value)
    return unicode_character

    # unicode_code_point = int(hex_string, 16)
    # # Check if the character is a control character
    # if 0 <= unicode_code_point <= 31:
    #     return ("\\x{:02x}".format(unicode_code_point))
    # else:
    #     return (chr(unicode_code_point))

def block_to_matrix(hex_string):
    chunks = [hex_string[i:i+2] for i in range(0, len(hex_string), 2)]

    # Create a 4x4 matrix
    matrix = [[0] * 4 for _ in range(4)]

    # Fill the matrix column-wise
    for i, chunk in enumerate(chunks):
        row_index = i % 4
        col_index = i // 4
        matrix[row_index][col_index] = chunk

    return matrix

def matrix_to_block(matrix):
    block = ''
    cipher = ''

    for col in range(len(matrix)):
        for row in range(len(matrix)):
            cipher+=(convert_hex_to_string(matrix[row][col]))            
            block += matrix[row][col]
    return block, cipher

def left_shift(matrix, n):
    shift_matrix = matrix[n:] + matrix[:n]
    return shift_matrix

def right_shift(matrix, n):
    shift_matrix =  matrix[-n:] + matrix[:-n]
    return shift_matrix

def calc_xor(matrix1, matrix2):
    matrix1_int = [int(hex_val, 16) for hex_val in matrix1]
    matrix2_int = [int(hex_val, 16) for hex_val in matrix2]

    # Perform XOR operation element-wise
    result_int = [a ^ b for a, b in zip(matrix1_int, matrix2_int)]

    # Convert result integers back to hexadecimal
    result_hex = [hex(val)[2:].zfill(2) for val in result_int]
    return result_hex

def calc_xor_2(matrix1, matrix2):
    result_matrix = []
    for row_a, row_b in zip(matrix1, matrix2):
        result_row = []
        for elem_a, elem_b in zip(row_a, row_b):
            binary_a = int(elem_a, 16)
            binary_b = int(elem_b, 16)
            xor_result = binary_a^binary_b
            result_row.append(hex(xor_result)[2:].zfill(2))
        result_matrix.append(result_row)
    return result_matrix

def get_substiitute_matrix(matrix):
    for i in range(len(matrix)): 
        b_bin = BitVector(hexstring=matrix[i])
        b_int = b_bin.intValue()
        s_int = Sbox[b_int]
        s_bin = BitVector(intVal=s_int, size=8)
        matrix[i] = s_bin.get_bitvector_in_hex()
    return matrix

def get_inv_substiitute_matrix(matrix):
    for i in range(len(matrix)): 
        b_bin = BitVector(hexstring=matrix[i])
        b_int = b_bin.intValue()
        s_int = InvSbox[b_int]
        s_bin = BitVector(intVal=s_int, size=8)
        matrix[i] = s_bin.get_bitvector_in_hex()
    return matrix

def get_g(matrix, idx):
    left_shift_1 = left_shift(matrix, 1)
    substiitute_matrix = get_substiitute_matrix(left_shift_1)
    xor = calc_xor(substiitute_matrix, round_constants[idx])
    return xor

def calc_next_round_key(idx):
    columns = [[row[i] for row in round_keys[idx]] for i in range(4)]
    g = get_g(columns[3], idx)
    
    temp_matrix = []
    temp_matrix.append(calc_xor(columns[0],g))

    for i in range(3):
        temp_matrix.append(calc_xor(columns[i+1], temp_matrix[i]))
    
    round_keys.append([[row[i] for row in temp_matrix] for i in range(len(temp_matrix))])

def key_expansion(key):
    round_keys.clear()
    hex_key = convert_string_to_hex(key)
    round_keys.append(block_to_matrix(hex_key))
    for i in range(10):
        calc_next_round_key(i)

def aes_substitute_matrix(matrix):
    substitute_matrix = []
    for i in range(len(matrix)):
        substitute_matrix.append(get_substiitute_matrix(matrix[i]))
    return substitute_matrix

def aes_inv_substitute_matrix(matrix):
    inv_substitute_matrix = []
    for i in range(len(matrix)):
        inv_substitute_matrix.append(get_inv_substiitute_matrix(matrix[i]))
    return inv_substitute_matrix

def aes_shifted_row_matrix(matrix):
    shifted_row_matrix = []
    for i in range(len(matrix)):
        shifted_row_matrix.append(left_shift(matrix[i], i))
    return shifted_row_matrix

def aes_inv_shifted_row_matrix(matrix):
    inv_shifted_row_matrix = []
    for i in range(len(matrix)):
        inv_shifted_row_matrix.append(right_shift(matrix[i], i))
    return inv_shifted_row_matrix

def aes_mix_cols_matrix(matrix):
    n = len(matrix)
    mix_columns_matrix = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                curr = BitVector(hexstring=matrix[k][j])
                x = curr.gf_multiply_modular(Mixer[i][k], AES_modulus, 8)
                mix_columns_matrix[i][j] ^=  x.intValue()
    
    for i in range(len(mix_columns_matrix)):
        for j in range(len(mix_columns_matrix[i])):
            mix_columns_matrix[i][j] = hex(mix_columns_matrix[i][j])[2:].zfill(2)

    return mix_columns_matrix

def aes_inv_mix_cols_matrix(matrix):
    n = len(matrix)
    inv_mix_columns_matrix = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                curr = BitVector(hexstring=matrix[k][j])
                x = curr.gf_multiply_modular(InvMixer[i][k], AES_modulus, 8)
                inv_mix_columns_matrix[i][j] ^=  x.intValue()
    
    for i in range(len(inv_mix_columns_matrix)):
        for j in range(len(inv_mix_columns_matrix[i])):
            inv_mix_columns_matrix[i][j] = hex(inv_mix_columns_matrix[i][j])[2:].zfill(2)

    return inv_mix_columns_matrix

def perform_aes_encryption(hex_plaintext):

    curr_state_matrix = calc_xor_2(block_to_matrix(hex_plaintext), round_keys[0])

    for i in range(9):
        substitute_matrix = aes_substitute_matrix(curr_state_matrix)
        shifted_row_matrix = aes_shifted_row_matrix(substitute_matrix)
        mix_columns_matrix = aes_mix_cols_matrix(shifted_row_matrix)
        new_state_matrix = calc_xor_2(mix_columns_matrix, round_keys[i+1])
        curr_state_matrix = new_state_matrix

    
    substitute_matrix = aes_substitute_matrix(curr_state_matrix)
    shifted_row_matrix = aes_shifted_row_matrix(substitute_matrix)
    new_state_matrix = calc_xor_2(shifted_row_matrix, round_keys[10])

    hex_ciphertext, ciphertext = matrix_to_block(new_state_matrix)
    return hex_ciphertext, ciphertext

def aes_encryption(hex_plaintext):
    bytes_data = bytes.fromhex(hex_plaintext)
    chunks = [bytes_data[i:i+16] for i in range(0, len(bytes_data), 16)]
    hex_slices = [chunk.hex() for chunk in chunks]


    hex_ciphertext = ''
    ciphertext = ''

    for block in hex_slices:
        a,b = perform_aes_encryption(block)
        hex_ciphertext+=a
        ciphertext+=b
    
    return ciphertext, hex_ciphertext

def perform_aes_decryption(hex_ciphertext):
    curr_state_matrix = calc_xor_2(block_to_matrix(hex_ciphertext), round_keys[10])

    for i in range(9):
        inv_shifted_row_matrix = aes_inv_shifted_row_matrix(curr_state_matrix)
        inv_substitute_matrix = aes_inv_substitute_matrix(inv_shifted_row_matrix)
        new_state_matrix = calc_xor_2(inv_substitute_matrix, round_keys[10-i-1])
        inv_mix_columns_matrix = aes_inv_mix_cols_matrix(new_state_matrix)
        curr_state_matrix = inv_mix_columns_matrix
    
    inv_shifted_row_matrix = aes_inv_shifted_row_matrix(curr_state_matrix)
    inv_substitute_matrix = aes_inv_substitute_matrix(inv_shifted_row_matrix)
    new_state_matrix = calc_xor_2(inv_substitute_matrix, round_keys[0])

    hex_plaintext, plaintext = matrix_to_block(new_state_matrix)
    return hex_plaintext, plaintext

def aes_decryption(ciphertext):
    hex_ciphertext = convert_string_to_hex(ciphertext)
    bytes_data = bytes.fromhex(hex_ciphertext)
    chunks = [bytes_data[i:i+16] for i in range(0, len(bytes_data), 16)]
    hex_slices = [chunk.hex() for chunk in chunks]


    hex_plaintext = ''
    plaintext = ''

    for block in hex_slices:
        a,b = perform_aes_decryption(block)
        hex_plaintext+=a
        plaintext+=b
    
    plaintext, hex_plaintext = pkcs7_unpad(plaintext.encode('utf-8'), hex_plaintext)
    return plaintext.decode('utf-8'), hex_plaintext

 
if __name__=="__main__":
    key = "Two One Nine Two"
    plaintext = "Hello World"


    # key = input("\nenter your key : ")
    start_time = time.time()
    processed_key = process_aes_key(key)
    key_expansion(processed_key)
    key_scheduling_time = time.time() - start_time
    
    
    # plaintext = input("enter your plaintext : ")
    start_time = time.time()
    padded_message = pkcs7_pad(plaintext.encode('utf-8'), 16)
    ciphertext, hex_ciphertext = aes_encryption(convert_bytes_to_hex(padded_message))
    encryption_time = time.time() - start_time
    

    start_time = time.time()
    deciphertext, hex_deciphertext = aes_decryption(ciphertext)
    decryption_time = time.time() - start_time


    print(BOLD + UNDERLINE + "\nKey : \n" + RESET_FORMATTING)
    print("In ASCII : ", key)
    print("In HEX : ", convert_string_to_hex(key))

    print(BOLD + UNDERLINE + "\nPlain Text : \n" + RESET_FORMATTING)
    print("In ASCII : ", plaintext)
    print("In HEX : ", convert_string_to_hex(plaintext))

    print(BOLD + UNDERLINE + "\nCipher Text : \n" + RESET_FORMATTING)
    print("In ASCII : ", ciphertext)
    print("In HEX : ", hex_ciphertext)

    print(BOLD + UNDERLINE + "\nDecipher Text : \n" + RESET_FORMATTING)
    print("In ASCII : ", deciphertext)
    print("In HEX : ", hex_deciphertext)

    print(BOLD + UNDERLINE + "\nExecution Time : \n" + RESET_FORMATTING)
    print(f"Key Scheduling : {key_scheduling_time} sec")
    print(f"Encryption Time : {encryption_time} sec")
    print(f"Decryption Time : {decryption_time} sec\n")





