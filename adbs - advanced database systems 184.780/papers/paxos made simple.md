> Lamport, Leslie. "Paxos made simple." ACM SIGACT News (Distributed Computing Column) 32, 4 (Whole Number 121, December 2001) (2001): 51-58.

the core of paxos is the **”synod consensus algorithm”**. the consensus is applied to the state machine to build distributed systems.

## overview

_problem_

- a bunch of processes can propose values
- we only want a single value of those which got proposed to be chosen – and shared among all processes

_roles_

- keep in mind: each process can have one or more roles in the implementation
- proposers
- acceptors
- learners

_message passing_

- based on the non-byzantine model
- agents have arbitrary processing speed
- messages can take an arbitrary time to be delivered or even lost – we are allowed to duplicate them
- messages can never be corrupted
- may fail and restart any time – we need redundancy

## step 1: choosing a value

_definitions_

- proposals $(n,v)$
     - a proposal consists of a “**proposal number** $v$“ and a “**value** $n$“.
- accepting
     - a proposal can be accepted by an acceptor.
- choosing
     - a proposal is chosen, if it got accepted by the majority.

_possible solutions_

there are multiple ways to approach this problem:

- one or more proposers send a value to a **single acceptor**.
     - problem: failure of acceptor can lead to failure of system.
     - solution: multiple acceptors for fault tolerance.
- one or more proposers send a value to **multiple acceptors concurrently**.
     - problem: failure of acceptor can lead to no choice being made, because no majority is reached.
     - solution: requirement (2).

_requirements_

1. acceptors must accept the first proposal they receive.
      - (note: this is necessary so the system functions even with a single proposer)
      - 1a. acceptors can’t accept higher promise numbers if they agreed to it in a response to a `prepare()` request.
1. multiple proposals can be chosen, as long as they all have the same value.
      - 2a. all following proposals that get accepted must have the same value (implies 2).
      - 2b. all following proposals that get proposed must have the same value (implies 2a).
      - 2c. for all proposals $(n,v)$ there is a set $S$ consisting of the majority of acceptors. for each acceptor, at least one of the conditions below must apply:
           - $\forall s \in S:$ no proposal with a number smaller than $n$ was ever accepted.
           - $\forall s \in S:$ the accepted proposal with a number smaller than but closest to $n$, has the same value $v$.

_satisfiying requirements_

- `prepare()` request – which satisfies the (2c) requirement:
     1. the proposer selects a random proposal number $n$
           - a proposer can make multiple proposals and can also abandon a proposal in the middle of the protocol. but it can’t reuse this number for another proposal. proposers remember the highest proposal number they ever used.
           - no two proposals are ever issued with the same number.
     1. the proposer requests all acceptors not to accept any other proposals with a number smaller than $n$.
           - the acceptors promise not to, if the number they received is the highest one out of all proposal they’ve received so far.
     1. the proposer requests all acceptors to respond with the proposal they accepted with a number smaller than but closest to $n$.
- `accept()` request
     1. if the proposer receives a majority response, then it can **issue a proposal to be accepted** with $v$ being either:
           - a) the highest numbered proposal among all responses.
           - b) an arbitrary value (if responders didn’t return any proposal).
     2. the acceptors accept this proposal, unless they received a `prepare()` request with a higher number in the meantime.

one possible performance optimization could be letting proposers know when a proposal with a higher number has been found, so they immediately restart.

if proposers constantly send `prepare()` requests with increasing values, then no progress will ever be made (livelock). this can be avoided by electing one or many special proposers, called **”distinguished proposers”**, that manage the proposals by receiving, selecting and broadcasting them all to the acceptors – similar to a proxy.

## step 2: learning a chosen value

learners learn that a value has been chosen, by finding out that it was accepted by the majority of acceptors.

there are many ways to implement this:

- acceptors broadcasting their accepted value to learners.
- learners broadcasting a request to learn to all acceptors.
- acceptors broadcasting to a subset of learners which are responsible to then forwarding / broadcasting the request to all learners - called the **”distinguished learners”**.

## the state machine

in a simple client-server architecture the server can be described as a deterministic state-machine that performs client instructions by executing a series of steps.

state machines receive an input, update their internal state, return an output.

we want to guarantee that all servers execute the same sequence of state machines commands to then reach the same final state.

we run multiple instances of paxos and the $i^{th}$ instance will determine the $i^{th}$ state-machine instruction of a sequence.

each server plays the same role in all instances of the algorithm (optimization: electing leaders for each role in each instance).
