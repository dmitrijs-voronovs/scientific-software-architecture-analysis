tactic_descriptions = {
    "Ping/Echo": "An asynchronous request/response message pair exchanged between nodes to determine reachability and responsiveness.",
    "Monitor": "A component that monitors the state of health of various parts of the system such as processors, processes, I/O, and memory.",
    "Heartbeat": "A fault detection mechanism that employs periodic message exchange between a system monitor and a process being monitored.",
    "Timestamp": "Used to detect incorrect sequences of events by assigning the state of a local clock to events immediately after they occur.",
    "Sanity Checking": "Checks the validity or reasonableness of specific operations or outputs of a component.",
    "Condition Monitoring": "Involves checking conditions in a process or device to prevent a system from producing faulty behavior.",
    "Voting": "Employs multiple components that do the same thing with identical inputs and forwards their output to voting logic to detect inconsistencies.",
    "Exception Detection": "The detection of a system condition that alters the normal flow of execution.",
    "Self-Test": "Components can run procedures to test themselves for correct operation.",
    "Active Redundancy": "A configuration where all nodes receive and process identical inputs in parallel, allowing redundant spares to maintain synchronous state with active nodes.",
    "Passive Redundancy": "A configuration where only active members process input traffic and provide redundant spares with periodic state updates.",
    "Spare": "Cold sparing where redundant spares remain out of service until a fail-over occurs.",
    "Exception Handling": "Mechanisms employed to handle detected exceptions using information to mask the fault.",
    "Rollback": "Permits the system to revert to a previous known good state upon detection of a failure.",
    "Software Upgrade": "Achieves in-service upgrades to executable code images in a non-service-affecting manner.",
    "Retry": "Assumes that the fault causing a failure is transient and retrying the operation may lead to success.",
    "Ignore Faulty Behavior": "Calls for ignoring messages from a particular source when determined to be spurious.",
    "Degradation": "Maintains critical system functions in the presence of component failures by dropping less critical functions.",
    "Reconfiguration": "Recovers from component failures by reassigning responsibilities to remaining functional resources.",
    "Shadow": "Operates a previously failed component in \"shadow mode\" for a predefined duration before reverting it to an active role.",
    "State Resynchronization": "Ensures that failed components are brought back to a consistent state with active components.",
    "Escalating Restart": "Allows system recovery by varying the granularity of components restarted while minimizing service impact.",
    "Non-Stop Forwarding": "Splits functionality into control and data planes to continue operations while recovering the supervisory information.",
    "Removal from Service": "Temporarily placing a system component in an out-of-service state to mitigate potential system failures.",
    "Transactions": "Ensures that asynchronous messages exchanged between distributed components are atomic, consistent, isolated, and durable.",
    "Predictive Model": "Used with monitors to ensure a system operates within nominal parameters and take corrective action for conditions predictive of future faults.",
    "Exception Prevention": "Techniques employed to prevent system exceptions from occurring.",
    "Increase Competence Set": "Expanding the set of states in which a program is competent to operate to handle more cases as part of normal operation.",
    "Discover Service": "Locates a service through searching a known directory service at runtime.",
    "Orchestrate": "Uses a control mechanism to coordinate, manage and sequence the invocation of particular servicess that could be ignorant of each other.",
    "Tailor Interface": "Adds or removes capabilities to an interface, such as translation, buffering, or hiding particular functions from untrusted users.",
    "Split Module": "Refining a module into several smaller modules to reduce the average cost of future changes.",
    "Increase Semantic Coherence": "Moving responsibilities that don't serve the same purpose to different modules to reduce the likelihood of side effects.",
    "Encapsulate": "Localizes user interface responsibilities to a single place.",
    "Use an Intermediary": "Breaking a dependency between responsibilities by using an intermediary that depends on the type of dependency.",
    "Restrict Dependencies": "Restricting the modules that a given module interacts with or depends on through visibility or authorization.",
    "Refactor": "Factoring out common responsibilities from modules where they exist and assigning them an appropriate home to reduce duplication.",
    "Abstract Common Services": "Implementing similar servicess once in a more general (abstract) form to reduce modification costs.",
    "Component Replacement": "Binding values at compile time or build time through replacement in build scripts or makefiles.",
    "Compile-time Parameterization": "Binding values at compile time through parameterization.",
    "Aspects": "Binding values at compile time or build time using aspects.",
    "Configuration-time Binding": "Binding values at deployment time through configuration.",
    "Resource Files": "Binding values at startup or initialization time using resource files.",
    "Runtime Registration": "Binding values at runtime through registration.",
    "Dynamic Lookup": "Binding values at runtime through dynamic lookup for servicess.",
    "Interpret Parameters": "Binding values at runtime by interpreting parameters.",
    "Startup Time Binding": "Binding values at runtime during startup.",
    "Name Servers": "Binding values at runtime using name servers.",
    "Plug-ins": "Binding values at runtime through plug-ins.",
    "Publish-Subscribe": "Binding values at runtime using the publish-subscribe pattern.",
    "Shared Repositories": "Binding values at runtime through shared repositories.",
    "Polymorphism": "Binding values at runtime using polymorphism.",
    "Manage Sampling Rate": "Reduce the frequency at which environmental data is captured to decrease demand, typically with some loss of fidelity.",
    "Limit Event Response": "Process events only up to a set maximum rate to ensure more predictable processing when events are processed.",
    "Prioritize Events": "Impose a priority scheme that ranks events according to how important it is to service them.",
    "Reduce Overhead": "Co-locating resources and removing intermediaries and abstractions to reduce computational overhead and energy demands.",
    "Bound Execution Times": "Place a limit on how much execution time is used to respond to an event.",
    "Increase Resource Efficiency": "Improve the algorithms used in critical areas to decrease latency.",
    "Increase Resources": "Use faster processors, additional processors, additional memory, and faster networks to reduce latency.",
    "Introduce Concurrency": "Process requests in parallel to reduce the blocked time.",
    "Maintain Multiple Copies of Computations": "Use multiple servers in a client-server pattern as replicas of computation to reduce the contention.",
    "Maintain Multiple Copies of Data": "Keep copies of data on storage with different access speeds to reduce the contention from multiple simultaneous accesses.",
    "Bound Queue Sizes": "Control the maximum number of queued arrivals and consequently the resources used to process the arrivals.",
    "Schedule Resources": "Understand the characteristics of each resource's use and choose the scheduling strategy that is compatible with it.",
    "Detect Intrusion": "Comparison of network traffic or service request patterns within a system to a set of signatures or known patterns of malicious behavior stored in a database.",
    "Detect Service Denial": "Comparison of the pattern or signature of network traffic coming into a system to historic profiles of known denial-of-service attacks.",
    "Verify Message Integrity": "Employment of techniques such as checksums or hash values to verify the integrity of messages, resource files, deployment files, and configuration files.",
    "Detect Message Delay": "Detection of potential man-in-the-middle attacks by checking the time that it takes to deliver a message to identify suspicious timing behavior.",
    "Identify Actors": "Identifying the source of any external input to the system, typically through user IDs, access codes, IP addresses, protocols, and ports.",
    "Authenticate Actors": "Ensuring that an actor is actually who or what it purports to be through passwords, one-time passwords, digital certificates, and biometric identification.",
    "Authorize Actors": "Ensuring that an authenticated actor has the rights to access and modify either data or servicess through access control mechanisms.",
    "Limit Access": "Limiting access to computing resources such as memory, network connections, or access points by using memory protection, blocking a host, closing a port, or rejecting a protocol.",
    "Limit Exposure": "Minimizing the attack surface of a system by reducing the number of access points for resources, data, or servicess and connectors that may provide unanticipated exposure.",
    "Encrypt Data": "Protecting data from unauthorized access by applying some form of encryption to data and to communication.",
    "Separate Entities": "Separating different entities within the system through physical separation, virtual machines, air gaps, or separation of sensitive from nonsensitive data.",
    "Change Default Settings": "Forcing users to change default settings to prevent attackers from gaining access to the system through publicly available settings.",
    "Revoke Access": "Severely limiting access to sensitive resources when the system or administrator believes that an attack is underway.",
    "Lock Computer": "Limiting access from a particular computer if there are repeated failed attempts to access an account from that computer.",
    "Inform Actors": "Notifying relevant personnel or cooperating systems when the system has detected an attack.",
    "Maintain Audit Trail": "Keeping a record of user and system actions and their effects to help trace the actions of, and to identify, an attacker.",
    "Restore": "Restoration of servicess after an attack using tactics that deal with recovering from a failure.",
    "Specialized Interfaces": "Providing test-specific interfaces that allow testers to control or observe component variables and states that might otherwise be inaccessible.",
    "Record/Playback": "Capturing the state when it crosses an interface to allow that state to be used to \"play the system back\" and re-create faults.",
    "Localize State Storage": "Storing state in a single place to make it easier to set the system to an arbitrary state for testing.",
    "Abstract Data Sources": "Designing interfaces to easily substitute test data without changing functional code.",
    "Sandbox": "Isolating the system from the real world to enable experimentation without permanent consequences.",
    "Executable Assertions": "Placing code at strategic locations to indicate when a program is in a faulty state.",
    "Limit Structural Complexity": "Reducing dependencies between components, simplifying inheritance hierarchies, and increasing cohesion to make behavior more predictable and easier to test.",
    "Limit Nondeterminism": "Finding and eliminating sources of unpredictable behavior to make testing more reliable.",
    "Increase semantic coherence": "Localizes user interface responsibilities to a single place.",
    "Co-locate related responsibilities": "Localizes user interface responsibilities to a single place.",
    "Restrict dependencies": "Minimizes the ripple effect to other software when the user interface changes.",
    "Defer binding": "Lets you make critical user interface choices without having to recode.",
    "Cancel": "Allows the user to terminate a command with appropriate resource management and notification.",
    "Undo": "Maintains sufficient information about system state so that an earlier state may be restored at the user's request.",
    "Pause/resume": "Provides the ability to temporarily free resources so they may be reallocated to other tasks.",
    "Aggregate": "Allows operations to be applied to a group of objects, freeing the user from repetitive operations.",
    "Maintain task models": "Determines context so the system can have some idea of what the user is attempting and provide assistance.",
    "Maintain user models": "Explicitly represents the user's knowledge of the system to control response time and assistance.",
    "Maintain system models": "Determines expected system behavior so that appropriate feedback can be given to the user.",
    "Metering": "Collecting data about the energy consumption of computational devices via a sensor infrastructure in real time.",
    "Static Classification": "Statically classifying devices and computational resources based on benchmarking or reported device characteristics when real-time data collection is infeasible.",
    "Dynamic Classification": "Using dynamic models that take into consideration transient conditions to determine energy consumption when real-time data collection is infeasible.",
    "Vertical Scaling": "Adding or activating resources to meet processing demands, or removing/deactivating resources when demands no longer require them.",
    "Horizontal Scaling": "Adding additional servers, VMs, or resources to an existing pool for scaling up, or removing/idling such resources for energy efficiency.",
    "Scheduling": "Allocating tasks to computational resources to optimize energy usage while respecting task constraints and priorities.",
    "Brokering": "Matching service requests with service providers based on energy information to allow choosing providers based on their energy characteristics.",
    "Service Adaptation": "Dynamically switching computational resources to ones that offer better energy efficiency or lower energy costs.",
    "Increase Efficiency": "Improving the time or memory performance of critical algorithms to enhance energy efficiency, or matching service requests to hardware best suited for those requests."}
