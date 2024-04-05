import random
import time

BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET_FORMATTING = "\033[0m"


def calc_r_m(n):
    i=1
    while True:
        val = n/pow(2, i)
        if(val.is_integer()):
            i+=1
        else:
            return i-1, int(n/pow(2, i-1))
        
def check_prime(a, m, n):
    x = pow(a, m, n)
 
    if (x == 1 or x == n - 1):
        return True
 
    while (m != n - 1):
        x = (x * x) % n
        m *= 2
 
        if (x == 1):
            return False
        if (x == n - 1):
            return True
 
    return False
        
def miller_rabin(n, k=5):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # Write n as d * 2^r + 1
    r,m = calc_r_m(n-1)

    # Repeat the test with different random witnesses
    for _ in range(k):
        a = 2 + random.randint(1, n - 4)
        if(not check_prime(a, m, n)):
            return False
    return True

def generate_prime(bits):
    while True:
        num = random.getrandbits(bits)
        if len(bin(num)[2:]) == bits and miller_rabin(num):
            return num
        
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def mod_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0

def generate_key_pairs(K):
    p = generate_prime(K//2)
    q = generate_prime(K//2)

    while p == q:
        q = generate_prime(K//2)

    # print(f"Prime Numbers: {p, q}")

    n = p * q
    phi = (p - 1) * (q - 1)

    # Choose public key e such that 1 < e < phi and gcd(e, phi) = 1
    e = random.randint(2, phi - 1)
    while gcd(e, phi) != 1:
        e = random.randint(2, phi - 1)

    # Calculate private key d such that d*e ≡ 1 (mod phi)
    d = mod_inverse(e, phi)

    return ((e, n), (d, n))

def encrypt(plaintext, public_key):
    e, n = public_key
    ciphertext = []

    for char in plaintext:
        unicode_value = ord(char)  # Convert character to Unicode
        encrypted_value = pow(unicode_value, e, n)  # Encrypt Unicode value
        ciphertext.append(encrypted_value)
        
    return ciphertext

def decrypt(ciphertext, private_key):
    d, n = private_key
    plaintext = ""

    for char in ciphertext:
        decrypted_value = pow(char, d, n)  # Decrypt encrypted value
        decrypted_char = chr(decrypted_value)  # Convert decrypted value to character
        plaintext += decrypted_char  # Append decrypted character to plaintext

    return plaintext


# Main function
if __name__ == "__main__":
    k = int(input("\nBit Size : "))
    plaintext = input("Plaintext : ")
    
    start_time = time.time()
    public_key, private_key = generate_key_pairs(k)
    key_generation_time = time.time() - start_time

    start_time = time.time()
    ciphertext = encrypt(plaintext, public_key)
    encryption_time = time.time() - start_time

    start_time = time.time()
    decrypted_text = decrypt(ciphertext, private_key)
    decryption_time = time.time() - start_time


    print(BOLD + UNDERLINE + "\nPublic Key : " + RESET_FORMATTING, end='')
    print(f"(e, n): {public_key}")

    print(BOLD + UNDERLINE + "Private Key : " + RESET_FORMATTING, end='')
    print(f"(d, n): {private_key}")

    print(BOLD + UNDERLINE + "\nPlain Text : " + RESET_FORMATTING)    
    print(plaintext)
    
    print(BOLD + UNDERLINE + "\nEncrypted Test (ASCII) : " + RESET_FORMATTING)    
    print(ciphertext)

    print(BOLD + UNDERLINE + "\nDecrypted Text : " + RESET_FORMATTING)    
    print(decrypted_text)

    print(BOLD + UNDERLINE + "\nExecution Time : " + RESET_FORMATTING)  
    print(f"Key Generation Time : {key_generation_time} seconds")
    print(f"Encryption Time : {encryption_time} seconds")
    print(f"Decryption Time : {decryption_time} seconds")
