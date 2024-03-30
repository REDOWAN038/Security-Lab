import time
from BitVector_Helper import *

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
def convert_string_to_hex(str):
    bytes_object = str.encode('utf-8')
    hex_representation = bytes_object.hex()
    return hex_representation

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

def left_shift(matrix, n):
    shift_matrix = matrix[1:] + [matrix[0]]
    return shift_matrix

def calc_xor(matrix1, matrix2):
    matrix1_int = [int(hex_val, 16) for hex_val in matrix1]
    matrix2_int = [int(hex_val, 16) for hex_val in matrix2]

    # Perform XOR operation element-wise
    result_int = [a ^ b for a, b in zip(matrix1_int, matrix2_int)]

    # Convert result integers back to hexadecimal
    result_hex = [hex(val)[2:].zfill(2) for val in result_int]
    return result_hex

def get_substiitute_matrix(matrix):
    for i in range(len(matrix)): 
        b_bin = BitVector(hexstring=matrix[i])
        b_int = b_bin.intValue()
        s_int = Sbox[b_int]
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
    round_keys.append(block_to_matrix(key))
    for i in range(10):
        calc_next_round_key(i)

def perform_aes(plaintext, key):
    key_expansion(key)
    # state_matrix = block_to_matrix(plaintext)
    # round_key = block_to_matrix(key)

    # for row in state_matrix:
    #     print(row)
    # print("\n")
    # for row in round_key:
    #     print(row)
 
# key = input("enter your key : ")
key = "Thats my Kung Fu"
# print("key : ", key)

hex_key = convert_string_to_hex(key)
# print("key in hex : ", hex_key)

 
# plaintext = input("enter your plaintext : ")
plaintext = "Two One Nine Two"
# print("plaintext : ", plaintext)

hex_plaintext = convert_string_to_hex(plaintext)
# print("plaintext in hex : ", hex_plaintext)

# for i in range(10):
perform_aes(hex_plaintext, hex_key)