from data.twitter import data
metadata, idx_q, idx_a = data.load_data(PATH='data/twitter/')                   # Twitter

print(metadata)
print(idx_q)
print(idx_a)