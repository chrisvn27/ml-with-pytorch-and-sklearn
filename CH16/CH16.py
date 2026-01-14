import torch 

sentence = torch.tensor(
    [0,
     7,
     1,
     2,
     5,
     6,
     4,
     3]
)

torch.manual_seed(123)
embed = torch.nn.Embedding(10,16)
embedded_sentence = embed(sentence).detach()
print(embedded_sentence.shape)

omega = torch.empty(8,8)

for i, x_i in enumerate(embedded_sentence):
    for j, x_j in enumerate(embedded_sentence):
        omega[i,j] = torch.dot(x_i,x_j)

#More efficient
omega_mat = embedded_sentence.matmul(embedded_sentence.T)

print(torch.allclose(omega_mat, omega))
print("omega shape",omega.shape)
import torch.nn.functional as F
attention_weights = F.softmax(omega, dim=1)
print("attention weights shape",attention_weights.shape)
print('Sum across columns: ', torch.sum(attention_weights, dim=1))

x_2 = embedded_sentence[1, :] #Second input word
context_vec_2 = torch.zeros(x_2.shape)
for j in range(8):
    x_j = embedded_sentence[j,:]
    context_vec_2 += attention_weights[1, j] * x_j

print(context_vec_2)

#More efficiently
context_vectors = torch.matmul(attention_weights, embedded_sentence)

print(torch.allclose(context_vectors[1], context_vec_2))

# query, key, and value
torch.manual_seed(123)
d = embedded_sentence.shape[1]
U_query = torch.rand(d,d)
U_key = torch.rand(d,d)
U_value = torch.rand(d,d)

x_2 = embedded_sentence[1,:]
query_2 = U_query.matmul(x_2)
key_2 = U_key.matmul(x_2)
value_2 = U_value.matmul(x_2)

# We also need the key and value sentences for all other input
# elements, which we can compute as follows
keys = U_key.matmul(embedded_sentence.T).T
queries = U_query.matmul(embedded_sentence.T).T
values = U_value.matmul(embedded_sentence.T).T

#Checking if they are correct
print('query 2', torch.allclose(query_2, queries[1]))
print('key 2', torch.allclose(key_2, keys[1]))
print('value 2', torch.allclose(value_2, values[1]))

omega_23 = query_2.dot(keys[2])

omega_2 = query_2.matmul(keys.T)
print(omega_2)

attention_weights_2 = F.softmax(omega_2/ d**0.5, dim=0)
print(attention_weights_2)

context_vec_2 = attention_weights_2.matmul(values)
print(context_vec_2)

# Attention is all we need: introducing the orignal transformer
# architecture

torch.manual_seed(123)
d = embedded_sentence.shape[1]
one_U_query = torch.rand(d, d)

# Assuming we have eight attention heads
h= 8 
multihead_U_query = torch.rand(h, d, d)
multihead_U_key = torch.rand(h, d, d)
multihead_U_value = torch.rand(h, d, d)

# The computation involving the query projection for the ith
# data point in the jth head can be written as wollos
# {q^(i)}_j = U_{qj} x^(i)

multihead_U_query_2 = multihead_U_query.matmul(x_2)
print(multihead_U_query_2.shape)

# calculating key and value for each head
multihead_U_key_2 = multihead_U_key.matmul(x_2)
multihead_U_value_2 = multihead_U_value.matmul(x_2)

print(multihead_U_key_2[2])
print(multihead_U_key.shape)
print(embedded_sentence.shape)
multihead_keys = multihead_U_key.matmul(embedded_sentence.T)
print(multihead_keys.shape)
multihead_keys = multihead_keys.permute(0,2,1)

# Head 3, datapoint 2
print(multihead_keys.shape)
print(multihead_keys[2,1])

# Implementing for values

multihead_values = multihead_U_value.matmul(embedded_sentence.T)
multihead_values = multihead_values.permute(0 , 2, 1)

multihead_z_2 = torch.rand(8, 16)

# Concatenation and squashing (pag 557)
linear = torch.nn.Linear(8*16, 16)
context_vector_2 = linear(multihead_z_2.flatten())
print(context_vector_2.shape)

