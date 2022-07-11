
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Distiller(nn.Module):
    def __init__(self, student, teacher, momentum):
        super(Distiller, self).__init__()
        self.momentum = momentum

        self.student = student
        self.teacher = teacher
        
        self.renew_teacher()# init teacher model 

    def forward(self, inputs, temperature):

        if self.training:
            self.student.train()
            # self.teacher.train()
            self.teacher.train()
            s_outputs, s_features, s_activation_rates , s_gate_logits = self.student(inputs, temperature=1)
            # torch.cat(s_gate_logits, dim=1).shape

            with torch.no_grad():
                self.momentum_update(self.momentum)
                t_outputs, t_features, t_activation_rates, t_gate_logits = self.teacher(inputs, temperature=1)
            
            s_gate_logits = torch.cat(s_gate_logits, dim=0)  #(batch, num_router)
            t_gate_logits = torch.cat(t_gate_logits, dim=0)

            # print("s",s_outputs[0])

            # print("t",t_outputs[0])

            return [s_outputs, s_features, s_activation_rates , s_gate_logits], \
                        [t_outputs, t_features, t_activation_rates, t_gate_logits]
        else:
            self.student.eval()
            s_outputs, s_features, s_activation_rates , s_gate_logits = self.student(inputs, temperature=1)
            return s_outputs, s_features, s_activation_rates , s_gate_logits

    @torch.no_grad()
    def renew_teacher(self,):
        print("renew")
        self.momentum_update(0.)

    @torch.no_grad()
    def momentum_update(self, momentum):
        '''
        Update the teacher_encoder parameters through the momentum update:
        key_params = momentum * key_params + (1 - momentum) * query_params
        '''
        # For each of the parameters in each encoder
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data = p_t.data * momentum + p_s.detach().data * (1. - momentum)
