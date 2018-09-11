-------------------------------------------------------------------------------
-- generate synthetic cause-effect pairs using neural networks 
--
-- Copyright (c) 2016 Facebook (David Lopez-Paz)
-- 
-- All rights reserved.
-- 
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
-- 
-- 1. Redistributions of source code must retain the above copyright
--    notice, this list of conditions and the following disclaimer.
-- 
-- 2. Redistributions in binary form must reproduce the above copyright
--    notice, this list of conditions and the following disclaimer in the
--    documentation and/or other materials provided with the distribution.
-- 
-- 3. Neither the names of NEC Laboratories American and IDIAP Research
--    Institute nor the names of its contributors may be used to endorse or
--    promote products derived from this software without specific prior
--    written permission.
-- 
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
-- AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
-- IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
-- ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
-- LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
-- CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
-- SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
-- INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
-- CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
-- ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
-- POSSIBILITY OF SUCH DAMAGE.
-- 
-------------------------------------------------------------------------------

local nn = require 'nn'
local xlua = require 'xlua'
local pl_data = require 'pl.data'
local pl_string = require 'pl.stringx'
local rk = require 'randomkit'

local function normalize(x)
  local y = x:clone()
  local m = x:mean(1):mul(-1)
  local s = x:std(1):add(1e-3)
  y:add(m:expandAs(y)):cdiv(s:expandAs(y))
  return y
end

local function matrix2file(matrix, fname)
  local out = io.open(fname, "w")
  for i=1,matrix:size(1) do
      for j=1,matrix:size(2) do
          out:write(matrix[i][j])
          if j == matrix:size(2) then
              out:write("\n")
          else
              out:write(" ")
          end
      end
  end
  out:close()
end

local function random_randomkit(n)
  -- all randomkit, except hypergeometric and triangular
  local x = torch.randn(n,1)
  local t = torch.random(0,28)
  if (t ==  0) then return torch.randn(n):bernoulli(torch.uniform()) end
  if (t ==  1) then return rk.beta(x, torch.uniform(0.1,10), torch.uniform(0.1,10)) end
  if (t ==  2) then return rk.binomial(x, torch.random(1,100), torch.uniform(0,1)) end
  if (t ==  3) then return rk.chisquare(x, torch.random(1,10)) end
  if (t ==  4) then return rk.exponential(x, torch.uniform(0.1,2)) end
  if (t ==  5) then return rk.f(x, torch.random(1,20), torch.random(1,20)) end
  if (t ==  6) then return rk.gamma(x, torch.uniform(1,10), torch.uniform(1,10)) end
  if (t ==  7) then return rk.gauss(x) end
  if (t ==  8) then return rk.geometric(x, torch.uniform(0,1)) end
  if (t ==  9) then return rk.gumbel(x, torch.uniform(0,10), torch.uniform(0.1,10)) end
  if (t == 10) then return rk.interval(x,torch.uniform(0,20)) end
  if (t == 11) then return rk.laplace(x,torch.uniform(0,10), torch.uniform(0,5)) end
  if (t == 12) then return rk.logistic(x,torch.uniform(0,10), torch.uniform(0,5)) end
  if (t == 13) then return rk.lognormal(x,torch.uniform(0,10), torch.uniform(0,3)) end
  if (t == 14) then return rk.logseries(x,torch.uniform(0,1)) end
  if (t == 15) then return rk.negative_binomial(x, torch.random(1,20), torch.uniform(0,1)) end
  if (t == 16) then return rk.noncentral_chisquare(x, torch.random(1,5), torch.uniform(1,5)) end
  if (t == 17) then return rk.noncentral_f(x, torch.random(1,20), torch.random(1,20), torch.uniform(1,5)) end
  if (t == 18) then return rk.pareto(x, torch.uniform(1,5)) end
  if (t == 19) then return rk.poisson(x, torch.uniform(1,10)) end
  if (t == 20) then return rk.power(x, torch.uniform(1,5)) end
  if (t == 21) then return rk.rayleigh(x, torch.uniform(0.1,5)) end
  if (t == 22) then return rk.standard_cauchy(x) end
  if (t == 23) then return rk.standard_t(x, torch.random(1,20)) end
  if (t == 24) then return rk.uniform(x,torch.uniform(0,10), torch.uniform(0,10)) end
  if (t == 25) then return rk.vonmises(x,torch.uniform(0,10), torch.uniform(0.1,10)) end
  if (t == 26) then return rk.wald(x,torch.uniform(0,5), torch.uniform(0.1,5)) end
  if (t == 27) then return rk.weibull(x, torch.uniform(1,5)) end
  if (t == 28) then return rk.zipf(x, torch.uniform(1,5)) end
end

local function cause(n)
  local res = torch.randn(n,1)
  -- once in a while, allow for non-GMM causes
  if (torch.bernoulli(0.25) == 1) then
    res = random_randomkit(n)
  else
    local k     = torch.random(1,5)
    local pm    = torch.uniform(0,5)
    local pv    = torch.uniform(0,5)
    local means = torch.randn(k)*pm
    local stds  = ((torch.randn(k)+1)*pv):abs():sqrt()
    local ws    = torch.randn(k):abs()
    ws:div(ws:sum())
    local idx   = torch.multinomial(ws,n,true)
    res:cmul(stds:index(1,idx)):add(means:index(1,idx))
  end
  return normalize(res:view(-1,1))
end

local function effect(x,h)
  local z = torch.randn(x:size(1),5):mul(torch.uniform(0,0.5))
  local u = torch.cat(x,z,2)
  local h = h or 128

  local net = nn.Sequential()
  net:add(nn.Linear(u:size(2),h))
  net:add(nn.ReLU())
  net:add(nn.Linear(h,1))
  return normalize(net:forward(u)):clone()
end

local function pair(n)
  local x = cause(n)
  local y = effect(x)
  if (torch.bernoulli(0.5) == 0) then
    local y2 = effect(x)
    return x,y,y2
  else
    local z  = effect(y)
    local z2 = effect(y)
    return y,z,z2
  end
end

local function synset(N, n, dir)
    local meta = io.open(dir .. '/pairmeta.txt', "w")
    for i=1,N,3 do
      local x_i, y_i, y2_i = pair(n)
      -- X causes Y
      matrix2file(torch.cat(x_i,y_i,2), dir .. '/pair' .. i .. '.txt')
      meta:write(i .. ' ' .. '1\n')
      -- Y causes X
      matrix2file(torch.cat(y_i,x_i,2), dir .. '/pair' .. i+1 .. '.txt')
      meta:write(i+1 .. ' ' .. '2\n')
      -- X and Y are confounded
      if(torch.bernoulli(0.5) == 0) then
        matrix2file(torch.cat(y_i,y2_i,2), dir .. '/pair' .. i+2 .. '.txt')
        meta:write(i+2 .. ' ' .. '3\n')
      else
        matrix2file(torch.cat(y2_i,y_i,2), dir .. '/pair' .. i+2 .. '.txt')
        meta:write(i+2 .. ' ' .. '3\n')
      end
      xlua.progress(i,N)
    end
    meta:close()
end

local cmd = torch.CmdLine()
cmd:option('--dir','.','output dir')
cmd:option('--N',10,'how many synthetic examples?')
cmd:option('--n',1024,'how many samples in the synthetic examples?')
local params = cmd:parse(arg)

synset(params.N, params.n, params.dir)
