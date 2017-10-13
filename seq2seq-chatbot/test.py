from data.twitter import data
metadata, idx_q, idx_a = data.load_data(PATH='data/twitter/')                   # Twitter

# from data.cornell_corpus import data
# metadata, idx_q, idx_a = data.load_data(PATH='data/cornell_corpus/')          # Cornell Moive

x = idx_q
y = idx_a
ratio = [0.7, 0.15, 0.15]

# number of examples
data_len = len(x)
print(data_len)

lens = [ int(data_len*item) for item in ratio ]
print(data_len*0.7)
print(data_len*0.15)
print(data_len*0.15)

print(lens)

trainX, trainY = x[:lens[0]], y[:lens[0]]
print(x)
print()
print(y)
print()
print(x[:lens[0]])
print()
print(y[:lens[0]])
print()

testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
validX, validY = x[-lens[-1]:], y[-lens[-1]:]