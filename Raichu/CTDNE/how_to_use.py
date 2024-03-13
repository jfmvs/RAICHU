from tcn import TCN

encoder = TCN(self.hidden_size, 
                        dilations = [1, 2], 
                        use_skip_connections=False,
                        dropout_rate=self.dropout_rate,
                        kernel_size = 3,
                        nb_stacks = 1,
                        return_sequences=True, 
                        name = 'word_encoder')(previous_layer)