#! -*- coding: utf-8 -*-
from keras.layers import *
from keras import backend as K
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Model
max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 128
dmem = 32
emb =128
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


class My_LSTM(Layer):

    def __init__(self, units, **kwargs):
        self.units = units # 输出维度
        super(My_LSTM, self).__init__(**kwargs)

    def build(self, input_shape): # 定义可训练参数

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.units *4 ),
                                      initializer='glorot_normal',
                                      trainable=True)

        self.recurrent_kernel = self.add_weight(
                                shape=(self.units, self.units * 4),
                                name='recurrent_kernel',
                                initializer='glorot_normal',
                                trainable = True
                                                )
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]


    def step_do(self, step_in, states): # 定义每一步的迭代

        x_i = K.dot(step_in, self.kernel_i)
        x_f = K.dot(step_in, self.kernel_f)
        x_c = K.dot(step_in, self.kernel_c)
        x_o = K.dot(step_in, self.kernel_o)
        h_tm1= states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        i = K.hard_sigmoid(x_i + K.dot(h_tm1,self.recurrent_kernel_i))
        f = K.hard_sigmoid(x_f + K.dot(h_tm1,self.recurrent_kernel_f))
        o = K.hard_sigmoid(x_o + K.dot(h_tm1, self.recurrent_kernel_o))
        m =x_c + K.dot(h_tm1,self.recurrent_kernel_c)
        # c =  K.tanh(f * c_tm1 + i * m)
        # h = o *c
        c = f * c_tm1 + i * m

        h =  o * K.tanh(c)
        ch = K.concatenate([c,h])

        return ch, [h,c]

    def call(self, inputs):
        init_states = [K.zeros((K.shape(inputs)[0],self.units)),K.zeros((K.shape(inputs)[0],self.units))]
        outputs = K.rnn(self.step_do, inputs, init_states)
        return outputs[1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.units*2)



class MGM(Layer):

    def __init__(self,dmem,**kwargs):
        self.output_dim = dmem
        super(MGM,self).__init__(**kwargs)

    def build(self, input_shape):  # 定义可训练参数
        self.W_Du = self.add_weight(name='W1',
                                       shape=(input_shape[-1], self.output_dim),
                                       initializer='glorot_normal',
                                       trainable=True)

        self.W_Dr1 = self.add_weight(name='W2',
                                       shape=(input_shape[-1], self.output_dim),
                                       initializer='glorot_normal',
                                       trainable=True)

        self.W_Dr2 = self.add_weight(name='W3',
                                       shape=(input_shape[-1], self.output_dim),
                                       initializer='glorot_normal',
                                       trainable=True)


        self.b_Du = self.add_weight(name='b1',
                                    shape=(self.output_dim,),
                                    initializer='glorot_normal',
                                    trainable=True)
        self.b_Dr1 = self.add_weight(name='b2',
                                    shape=(self.output_dim,),
                                    initializer='glorot_normal',
                                    trainable=True)
        self.b_Dr2 = self.add_weight(name='b3',
                                    shape=(self.output_dim,),
                                    initializer='glorot_normal',
                                    trainable=True)

    def step_do(self,step_in,states):
        r1 = K.softmax(K.dot(step_in,self.W_Dr1) + self.b_Dr1)#公式10
        r2 = K.softmax(K.dot(step_in,self.W_Dr2) + self.b_Dr2)
        u_tide = K.dot(step_in,self.W_Du)+self.b_Du # 公式 9
        step_out = r1*states[0]+r2*K.tanh(u_tide)# 公式11

        return step_out,[step_out]


    def call(self,inputs):
        init_states = [K.zeros((K.shape(inputs)[0], self.output_dim))]
        outputs = K.rnn(self.step_do, inputs, init_states)
        return outputs[0]


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

print('Build model...')



S_inputs = Input(shape=(maxlen,), dtype='int32')
embedding= Embedding(max_features, emb)(S_inputs)
ch= My_LSTM(emb)(embedding)

c_s =Lambda(lambda x: x[:,:,:emb])(ch)#公式 5
h = Lambda(lambda x: x[:,:,emb:])(ch)#公式 6


c_s_2 = concatenate([c_s,c_s])
c_s_3 = concatenate([c_s_2,c_s])

t = Lambda(lambda x: x[:, :(maxlen-1)])(c_s_3)
t_1 = Lambda(lambda x: x[:, 1:])(c_s_3)
c = concatenate([t, t_1])


a = TimeDistributed( Dense(emb*6,activation='softmax'))(c) #公式 7
c_tide = multiply([c, a]) #公式8

u =  MGM(dmem)(c_tide)# 公式9-11

h2 = concatenate([h,h])
h3 = concatenate([h2,h])
h3 = Lambda(lambda x: x[:,-1,:])(h3)

final = concatenate([h3,u]) # Output of MFN


O_seq = Dense(1,activation='sigmoid')(final)
model = Model(inputs= S_inputs,outputs=O_seq)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))
