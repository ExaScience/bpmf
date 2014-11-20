type ReplicatedData{T}
  pids::Vector{Int}
  refs::Array{RemoteRef}

  pidx::Int
  val::T

  ReplicatedData(p,r) = new(p,r)
end

function ReplicatedData{T}(val::T; pids=Int[])
  if isempty(pids)
    pids = procs()
  end

  refs = Array(RemoteRef, length(pids))
  for (i,p) in enumerate(pids)
    refs[i] = remotecall(p, () -> val)
  end

  for i in 1:length(refs)
    wait(refs[i])
  end

  rd = ReplicatedData{T}(pids, refs)
  init_loc_flds(rd)
  rd
end

function init_loc_flds(rd::ReplicatedData)
  if myid() in rd.pids
    rd.pidx = findfirst(rd.pids, myid())
    rd.val = fetch(rd.refs[rd.pidx])
  else
    rd.pidx = 0
  end
end

import Base: serialize, deserialize, writetag, serialize_type, UndefRefTag

function serialize(s, rd::ReplicatedData)
  serialize_type(s, typeof(rd))
  for n in ReplicatedData.names
    if n in [:val, :pidx]
      writetag(s, UndefRefTag)
    else
      serialize(s, getfield(rd, n))
    end
  end
end

function deserialize{T}(s, t::Type{ReplicatedData{T}})
  rd = invoke(deserialize, (Any, DataType), s, t)
  init_loc_flds(rd)
  if rd.pidx == 0
    error("ReplicatedData cannot be used on a non-participating process")
  end
  rd
end
