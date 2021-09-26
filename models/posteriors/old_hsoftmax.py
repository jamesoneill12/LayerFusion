import torch
import torch.nn.functional as F



def latent_mixture(self, x, w_top, b_top, w_bottom, b_bottom):
    """No labels are given"""
    xs = torch.unsqueeze(x, dim=1)
    x = torch.cat([torch.matmul(xs, w_top[i]) + b_top[i] for i in range(self.mix_num)], 2)
    x = x.view(x.size(0), x.size(1), int(x.size(2) / self.mix_num), self.mix_num)
    if self.p is not None:
        x = x * F.softmax(self.p)
        layer_top_probs = x.sum(dim=3)
    else:
        layer_top_probs = F.softmax(torch.mean(x, 3))

    for j in range(self.ntokens_per_class):

        """
        layer_top_probs = self.softmax(layer_top_logits[:, j].unsqueeze(1))
        bottom_logits = torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0]
        word_probs = self.softmax(bottom_logits) * layer_top_probs # [:, 0]
        """

        bottom_logits = torch.cat([torch.matmul(xs, w_bottom[0]) + b_bottom[0] for i in range(self.mix_num)], 2)
        bottom_logits = x.view(bottom_logits.size(0), bottom_logits.size(1),
                               int(bottom_logits.size(2) / self.mix_num), self.mix_num)
        if self.p is not None:
            bottom_logits = bottom_logits * F.softmax(self.p)
            layer_bottom_probs = bottom_logits.sum(dim=3)
        else:
            layer_bottom_probs = F.softmax(torch.mean(bottom_logits, 3))
        word_probs = layer_bottom_probs * layer_top_probs  # [:, 0]

        for i in range(1, self.nclasses):
            bottom_logits = torch.cat([torch.matmul(xs, w_bottom[i]) + b_bottom[i] for i in range(self.mix_num)], 2)
            bottom_logits = x.view(bottom_logits.size(0), bottom_logits.size(1),
                                   int(bottom_logits.size(2) / self.mix_num), self.mix_num)
            temp = F.softmax(torch.mean(bottom_logits, 3))
            word_probs = torch.cat((word_probs, layer_top_probs * temp), dim=1)
    return word_probs


    def forward_mixture(self, inputs, labels=None):

        batch_size, d = inputs.size()

        if labels is not None:
            label_position_top = labels / self.ntokens_per_class
            label_position_bottom = labels % self.ntokens_per_class

            layer_top_probs = self.latent_mixture_labels(inputs, self.layer_top_W,
                  self.layer_top_b, labels = label_position_bottom, top=True)


            """
            CORRECT 
            print(inputs.size())
            print(self.layer_bottom_W.size())
            print(label_position_top.size())
            torch.Size([700, 50])
            torch.Size([10, 50, 10])
            torch.Size([700])            
            """

            layer_bottom_probs = self.latent_mixture_labels(inputs, self.layer_bottom_W,
                  self.layer_bottom_b, labels = label_position_bottom, top=False)
            # should output 700, 10

            """
            layer_top_probs.size() - torch.Size([700, 10])
            label_position_top.size() - torch.Size([700])
            layer_bottom_probs.size() - torch.Size([700, 10])
            label_position_bottom.size() - torch.Size([700])            
            """

            target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * \
                           layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]
            return target_probs

        else:

            word_probs = self.latent_mixture(inputs)
            """
            layer_top_probs = self.latent_mixture(inputs, self.layer_top_W, self.layer_top_b, top=True)
            for j in range(self.ntokens_per_class):
                word_probs = self.latent_mixture(inputs, self.layer_bottom_W[0], self.layer_bottom_b[0], top=False)
                for i in range(1, self.nclasses):
                    word_probs = torch.cat((word_probs, layer_top_probs * self.latent_mixture(
                        self.latent_mixture(inputs, self.layer_bottom_W[i]), self.layer_bottom_b[i], top=False)), dim=1) 
            """

            return word_probs