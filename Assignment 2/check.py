# hex_string = "29c3505f571420f6402299b31a02d73a"
hex_string = "c350"
text = bytes.fromhex(hex_string).decode("utf-8")
print(text)
