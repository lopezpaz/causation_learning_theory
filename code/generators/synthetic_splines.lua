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

require 'xlua'

local function spline(x,y,x0)
    local m   = torch.zeros(x0:size(1))
    local y0  = torch.zeros(x0:size(1))
    -- compute tangents
    for k=1,x:size(1)-1 do
        m[k] = (y[k+1]-y[k])/(x[k+1]-x[k])*.5
        m[k] = m[k] + (y[k]-y[k+1])/(x[k]-x[k+1])*.5
    end
    -- compute interpolations
    for i=1,x0:size(1) do
        -- find closest points
        for j=1,x:size(1)-1 do
            if ((x[j] <= x0[i]) and (x[j+1] >= x0[i])) then
                k = j
                break
            end
        end
        -- auxiliary variables
        local t = (x0[i]-x[k]) / (x[k+1] - x[k])
        local h00 = 2.*t*t*t - 3.*t*t + 1.
        local h10 = t*t*t - 2*t*t + t
        local h01 = -2.*t*t*t + 3.*t*t
        local h11 = t*t*t - t*t
        -- calculate y_i
        y0[i] = h00*y[k]
        y0[i] = y0[i]+h10*(x[k+1]-x[k])*m[k]
        y0[i] = y0[i]+h01*y[k+1]
        y0[i] = y0[i]+h11*(x[k+1]-x[k])*m[k+1]
    end
    return y0
end

local function cause(n)
    local k     = torch.random(1,5)
    local pm    = torch.uniform(0,5)
    local pv    = torch.uniform(0,5)
    local means = torch.rand(k)*pm
    local stds  = (torch.randn(k)*pv):abs()
    local ws    = torch.randn(k):abs()
    ws:div(ws:sum())
    local idx = torch.multinomial(ws,n,true)
    local res = torch.rand(n)
    res:cmul(stds:index(1,idx)):add(means:index(1,idx))
    res:add(-res:mean()):div(res:std())
    return res
end

local function effect(x)
    local d  = torch.random(4,5)
    local kx = torch.linspace(x:min()-x:std(),x:max()+x:std(),d)
    local ky = torch.randn(d)
    local y  = spline(kx,ky,x)
    y:add(-y:mean()):div(y:std())
    local v  = torch.uniform(0,5)
    local kv = torch.rand(d)*v
    local vv = spline(kx,kv,x)
    y:add(torch.randn(x:size(1)):cmul(vv))
    y:add(-y:mean()):div(y:std())
    return y
end

local function pair(n)
  local x = cause(n)
  local y = effect(x)
  if (torch.bernoulli() == 1) then
    local y2 = effect(x)
    return x,y,y2
  else
    local z  = effect(y)
    local z2 = effect(y)
    return y,z,z2
  end
end

function generate_batch(N, n)
    local result_x = {} 
    local result_y = {} 
    local result_w = {} 

    for i=1,N,6 do
      local x_i, y_i, y2_i = pair(n)
      local p_i = torch.randperm(n):long()
      -- X causes Y 
      result_x[#result_x+1] = torch.cat(x_i, y_i, 2)
      result_y[#result_y+1] = 1
      result_w[#result_w+1] = 1
      -- Y causes X 
      result_x[#result_x+1] = torch.cat(y_i, x_i, 2)
      result_y[#result_y+1] = 0
      result_w[#result_w+1] = 1
      -- X is independent from Y 
      result_x[#result_x+1] = torch.cat(x_i, y_i:index(1,p_i), 2)
      result_y[#result_y+1] = 0
      result_w[#result_w+1] = 1
      result_x[#result_x+1] = torch.cat(y_i:index(1,p_i), x_i, 2)
      result_y[#result_y+1] = 0
      result_w[#result_w+1] = 1
      -- X and Y are confounded 
      result_x[#result_x+1] = torch.cat(y_i, y2_i, 2)
      result_y[#result_y+1] = 0
      result_w[#result_w+1] = 1
      result_x[#result_x+1] = torch.cat(y2_i, y_i, 2)
      result_y[#result_y+1] = 0
      result_w[#result_w+1] = 1
      xlua.progress(i,N)
    end
    return result_x, result_y, result_w
end

local cmd = torch.CmdLine()
cmd:option('--N',1000,'number of scatterplots to generate')
cmd:option('--n',1000,'number of samples per scatterplot')
cmd:option('--output_file', 'synthetic.t7')
local params = cmd:parse(arg)

local result_x, result_y, result_w = generate_batch(params.N, params.n)
torch.save(params.output_file, { result_x, result_y, result_w })
