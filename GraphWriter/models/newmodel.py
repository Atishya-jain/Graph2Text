import torch
from torch import nn
from models.attention import MultiHeadAttention, MatrixAttn
from models.list_encoder import list_encode
from models.last_graph import graph_encode
from models.beam import Beam

class model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    cattimes = 3 if args.title else 2
    self.emb = nn.Embedding(args.ntoks,args.hsz)
    self.lstm = nn.LSTMCell(args.hsz*cattimes,args.hsz)
    self.out = nn.Linear(args.hsz*cattimes,args.tgttoks)
    self.list_encode = list_encode(args)
    self.switch = nn.Linear(args.hsz*cattimes,1)
    self.attn = MultiHeadAttention(args.hsz,args.hsz,args.hsz,h=4,dropout_p=args.drop)
    self.mattn = MatrixAttn(args.hsz*cattimes,args.hsz)
    self.graph = (args.model in ['graph','gat','gtrans'])
    print(args.model)
    if self.graph:
      self.graph_encode = graph_encode(args)

  def forward(self,batch):
    outp,_ = batch.out
    ents = batch.ent
    entlens = ents[2]
    ents = self.list_encode(ents)
    # print (ents.size())
    if self.graph:
      gents, glob, grels = self.graph_encode(batch.rel[0], batch.rel[1], (ents, entlens))
      hx = glob
      keys,mask = grels
      mask = mask==0
    mask = mask.unsqueeze(1)

    cx = torch.tensor(hx)
    a = torch.zeros_like(hx) #self.attn(hx.unsqueeze(1),keys,mask=mask).squeeze(1)
    e = self.emb(outp).transpose(0,1)
    outputs = []
    for i, k in enumerate(e):
      prev = torch.cat((a,k), 1)
      hx, cx = self.lstm(prev, (hx, cx))
      a = self.attn(hx.unsqueeze(1), keys, mask=mask).squeeze(1)
      out = torch.cat((hx, a), 1)
      outputs.append(out)
    l = torch.stack(outputs, 1)
    pgen = torch.sigmoid(self.switch(l))
    generate_dist = self.out(l)
    generate_dist = torch.softmax(generate_dist, 2)
    generate_dist = pgen*generate_dist

    #compute copy attn
    _, z = self.mattn(l,(ents,entlens))
    z = (1-pgen)*z
    o = torch.cat((generate_dist,z),2)
    o = o+(1e-6*torch.ones_like(o))

    return o.log(), z, None

  def maskFromList(self,size,l):
    mask = torch.arange(0,size[1]).unsqueeze(0).repeat(size[0],1).long().cuda()
    mask = (mask <= l.unsqueeze(1))
    mask = mask==0
    return mask
    
  def emb_w_vertex(self,outp,vertex):
    mask = outp>=self.args.ntoks
    if mask.sum()>0:
      idxs = (outp-self.args.ntoks)
      idxs = idxs[mask]
      verts = vertex.index_select(1,idxs)
      outp.masked_scatter_(mask,verts)

    return outp

  def beam_generate(self,batch,beamsz,k):
    ents = batch.ent
    entlens = ents[2]
    ents = self.list_encode(ents)
    if self.graph:
      gents,glob,grels = self.graph_encode(batch.rel[0],batch.rel[1],(ents,entlens))
      hx = glob
      #hx = ents.max(dim=1)[0]
      keys,mask = grels
      mask = mask==0
    mask = mask.unsqueeze(1)

    cx = torch.tensor(hx)
    a = self.attn(hx.unsqueeze(1),keys,mask=mask).squeeze(1)
    outputs = []
    outp = torch.LongTensor(ents.size(0),1).fill_(self.starttok)
    beam = None
    for i in range(self.maxlen):
      op = self.emb_w_vertex(outp.clone(),b.nerd)
      op = self.emb(op).squeeze(1)
      prev = torch.cat((a,op),1)
      hx,cx = self.lstm(prev,(hx,cx))
      a = self.attn(hx.unsqueeze(1),keys,mask=mask).squeeze(1)

      l = torch.cat((hx,a),1).unsqueeze(1)
      s = torch.sigmoid(self.switch(l))
      o = self.out(l)
      o = torch.softmax(o,2)
      o = s*o
      #compute copy attn
      _, z = self.mattn(l,(ents,entlens))
      #z = torch.softmax(z,2)
      z = (1-s)*z
      o = torch.cat((o,z),2)
      o[:,:,0].fill_(0)
      o[:,:,1].fill_(0)

      o = o+(1e-6*torch.ones_like(o))
      decoded = o.log()
      scores, words = decoded.topk(dim=2,k=k)
      if not beam:
        beam = Beam(words.squeeze(),scores.squeeze(),[hx for i in range(beamsz)],
                  [cx for i in range(beamsz)],[a for i in range(beamsz)],beamsz,k,self.args.ntoks)
        beam.endtok = self.endtok
        beam.eostok = self.eostok
        keys = keys.repeat(len(beam.beam),1,1)
        mask = mask.repeat(len(beam.beam),1,1)

        ents = ents.repeat(len(beam.beam),1,1)
        entlens = entlens.repeat(len(beam.beam))
      else:
        if not beam.update(scores,words,hx,cx,a):
          break
        keys = keys[:len(beam.beam)]
        mask = mask[:len(beam.beam)]
        ents = ents[:len(beam.beam)]
        entlens = entlens[:len(beam.beam)]
      outp = beam.getwords()
      hx = beam.geth()
      cx = beam.getc()
      a = beam.getlast()

    return beam