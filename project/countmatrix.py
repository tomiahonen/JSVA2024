with open('C:\\Users\\tomia\\OneDrive\\Työpöytä\\JSVA2024\\JSVA2024\\project') as f:
    data = f.read().splitlines()
    data = [list(map(int, x.split())) for x in data]

def count_matrix(data):
    count = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == 1:
                count += 1
    return count

print(count_matrix(data))
