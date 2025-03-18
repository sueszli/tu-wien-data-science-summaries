> Tanenbaum, A., & Van Steen, M. (2023). Distributed Systems. Amazon Digital Services LLC - Kdp.

# 1. introduction

_”distributed system” definitions_

- “\[… a system\] in which the failure of a computer you didn’t even know existed can render your own computer unusable”- leslie lamport
- “a distributed system is a collection of autonomous computing elements that appears to its users as a single coherent system” - tanenbaum
     1. nodes can act independently
     2. nodes look like a single system from the outside (= distribution transparency)
- that means it’s a bunch of geographically dispersed nodes (hardware device / software process) that communicate with eachother over a network to provide a service together. the goal is to make the nodes collaborate. this is done through message passing, a global clock, …
- the topology of the network (physical network / overlay network) changes all the time.
     - “open groups” let any node in, “closed groups” need a control mechanism.
     - “structured overlays” have a fixed topology, “unstructured overlays” just have random neighbours.

_middleware_

- = distributed-system layer
- = “middleware is the same to a distributed system as what an operating system is to a computer: a manager of resources offering its applications to efficiently share and deploy these resources across a network \[, with the difference being that it’s offered over a network\].”
- software layer on top of operating systems on multiple machines to achieve transparency.
- hides differences in hardware and operating system for each distributed application.
- can also serve as a library for convenience.
- services provide:
     - communication: rpcs, …
     - transactions: atomicity, …
     - service composition: web service api, …
     - reliability: guarantees for message passing, …

_design goals_

- making remote resource sharing easier
- hiding the fact that resources are distributed across a network (= transparency of access, location, relocation, migration, replication, concurrency, failure)
- being open, in a way that components are easily reusable and integratable into other systems
- being scalable by avoiding the 3 possible bottlenecks:
     - compute bound: computational capacity, limited by cpus – resolved by scaling up (vertical scaling, more memory and cpu power) or scaling out (horizontal scaling, more machines, parallel processing).
     - i/o bound: storage capacity, including i/o transfer rate – resolved through asynchronous processing.
     - network bound: the network between client and server not having enough bandwidth

_false assumptions_

- the network is reliable
- the network is secure
- the network is homogenous
- the topology doesn’t change
- latecncy is zero
- bandwidth is infinite
- transport cost is zero
- there is only one administrator

_classification_

- distributed computing systems
     - cluster computing = similar machines with the same operating system in high speed LAN, used for high performance computing. highest performance can be achieved when processors have their own seperate memory, don’t have to share, but have access through direct memory access (dma) in distributed shared memory multicomputers (dsm).
     - grid computing = different machines, specialized for specific tasks / equipped with special peripherals, used for high performance computing.
     - cloud computing = outsourcing the entire infrastructure for convenience.
          - haas / hardware as a service: entire datacenteres (cpu, memory disk, bandwidth, processors, routers, cooling systems)
          - iaas / infrastructure as a service: virtual computing resources (vm, aws ec2), virtual storage (blocks, files, aws s3)
          - paas / platforms as a service: convenient software frameworks for deployment and database apis, system level services (buckets, ms azure, google app engine)
          - saas / software applications as a service: web apps (google docs, gmail, youtube)
- distributed information systems
     - mostly integration software like: rpc / remote procedure call, rmi / remote method invocation (rpc but for objects instead of functions), mom / message passing middleware for publish/subscribe systems, transactional operations that are acid compliant in shared databases, file transfer
- distributed pervasive systems
     - ubiquitous computing = interactive and have some sort of context awareness and autonomy.
     - mobile computing = machines can change their position, communicate over a wireless network, ideally a manet / mobile adhoc network that’s distruption tolerant and uses flooding for message passing.
     - sensor networks = small nodes with peripherals as data sources. can be used for in-network data processing.

# 2. architectures

## 2.1. software architecture

_software architecture_

- = organization of software components
     - component (client, server, peer)
     - connector (rpc, mom)
     - interface (api)
     - protocol (http, tcp, udp)
- layered architectures = application layering (like the 3-tier architecture) or protocol layering (like the osi model)
- object-based architectures = oop patterns, objects that communicate through rmi
- service-oriented architecture (soa) = services that commuicate through rpc
- resource-centered architectures = restfulness
- event-based architectures = publish-subscribe systems with mom
     - topic-based: you can subscribe to specific event types called ‘topics’
     - content-based: you can additionally use sql-like queries to subscribe to topics
     - the sender doesn’t need to know the precise identity of the receiver but the communication is synchronous

_coordination_

- coupling through time and space:
     - temporally coupled / synchronous = both parties must be simultaneously available → can also be determined through the protocol (tcp, udp)
     - referentially coupled = the sender must know the precise identity of the receiver
- 4 possible combinations:
     - ref + sync = direct communication (rpc, rmi, restul api, …)
     - ref + async = mailbox communication (email, message queues, …)
     - no ref + sync = event-based communication → this is the publish-subscribe pattern
     - no ref + async = shared data space communication (tuples in database, shared memory, …)

_middleware design patterns_

- wrapper / adapter / proxy: makes interfaces compatible
- message broker: forwards messages to the correct destination
- interceptor: can interrupt processes to add additional functionality

## 2.2. system architecture

_system architecture_

- = distribution of software on physical machines
- centralized organization
     - simple client-server architecture
     - multi-tier architecture = software layers logically isolated and put on different machines
- decentralized organization
     - peers/servants: nodes that can act as both client and server
     - overlay network: logical network
          - structured overlay network = specific topology (ring, tree, mesh, …) to look up nodes (routing). lookup usually based on a dht / distributed hash table / semantic-free index. the system as a whole is responsible for the dht and usually each node stores a subset of the hash-table.
          - unstructured overlay network = edges in topology only exist with a certain probability. you need specific algorithms to look nodes up:
               - flooding: recursively broadcasting a search request.
               - random walks: flooding but but a random set of neighbours.
               - policy based search: flooding but with a list of preferred nodes, based on response-quality and time-to-live (ttl) parameter.
          - hierarchically organized overlay networks = “super peers” / “leaders” that get elected and manage the “weak peers”.
- hybrid organization (combination of both)
     - edge-server systems
          - edge: sits between local network / enterprise network and the public internet / isp service.
          - fog computing: sits between edge and the clients devices. one edge server acts as the “origin” from which others replicate and serve content or assist in computation.
     - collaborative distributed systems
          - usually file sharing systems like bittorrent: bittorrent users download file chunks from each other to assemble the complete file – but the downloading / leeching nodes are forced to share / seed to prevent free-riding. nodes join the system through a central server and access a global list with references to torrent files, containing links to tracker servers which keep track of active nodes downloading the requested file. a faster implementation uses a dht instead of a central tracking server.

# 3. processes

_definitions_

- **processor** = cpu, processing unit
- **core** = is inside the processor. popular because multicores are cheaper than multiprocessors.
- **process** = a program during execution. has its own process context (address space, code, data, resources).
- **thread** = is inside a process. more efficient than a process but also less secure / isolated.
     - **one-to-one threading / kernel level threads**:
          - os schedules threads directly.
          - expensive context switching, great scheduling and parallelism.
          - threads can be blocked in isolation.
          - very simple and therefore often preferred.
     - **many-to-one threading / user level threads**:
          - os schedules process containing threads.
          - cheap context switching, bad scheduling and parallelism.
          - when the process is blocked, then all threads are.
     - **many-to-many threading / light weight processes (lwp) with user level threads**:
          - lwps get managed in the user space and can run inside a single heavy process.
          - the threads get assigned to their own LWP after being generated.
     - **… other types**: green threads, protothreads, fibers, coroutines are all lightweight alternatives to threads, typically used to implement asynchronous operations. goroutines are special coroutines used in golang.

_non-blocking system call_

- single-threaded concurrency.
- a finite state machine used to reschedule the instruction order in a single thread instead of waiting for expensive os calls for i/o bound tasks.
- example: node.js event loop.

_resource virtualization_

- important for security/isolation, portability in cloud computing.
- mimicks an interface at different levels:
     - process virtual machine / runtime system = abstract instruction set, like java virtual machine or os emulators like wine.
     - native virtual machine = mimicking the isa (hardware instruction set architecture) or system calls. allows having multiple guest operating systems.

...
