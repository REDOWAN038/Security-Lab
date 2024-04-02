import random

def calc_r_m(n):
    i=1

    while True:
        val = n/pow(2, i)
        if(val.is_integer()):
            i+=1
        else:
            return i-1, int(n/pow(2, i-1))
        
def calc_prime(a, m, n):
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


def isPrime(n, k):
    if n<=1:
        return False
    if n<=3:
        return True
    if n%2==0:
        return False
    
    
    r,m = calc_r_m(n-1)
    a = 2 + random.randint(1, n - 4)
    res = calc_prime(a, m, n)
    d = n - 1
    while (d % 2 == 0):
        d //= 2

    print("r ", r)
    print("m ", m)
    print("d ", d)
    return res

print(isPrime(56999, 5))

# Prime Numbers: (6079, 56999)