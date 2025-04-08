from typing import Dict, List

QualityAttributesMap = Dict[str, List[str]]

quality_attributes = {
    "Energy Efficiency": ["energy", "power", "consump", "efficien", "battery", "charg", "drain", "watt", "joule",
                          "green", "sustainab", "meter", "monitor", "reduc", "allocat", "adapt", "schedul", "sensor",
                          ]}

quality_attributes_all = {
    **quality_attributes,
    "Availability": ["avail", "downtime", "outage", "reliab", "fault", "failure", "error", "robust", "toler",
                     "resilien", "recover", "repair", "failover", "fail-safe", "redundant", "mask", "degraded",
                     "mainten", "heartbeat", "ping", "echo", "rollback", "checkpoint", "reboot", "alive", "down",

                     ],
    "Deployability": ["deploy", "release", "update", "install", "rollout", "rollback", "upgrade", "integrat",
                      "continuous", "hotfix", "patch", "pipeline", "configurat", "rolling", "kill switch",
                      "feature toggle", "toggle", "canary", "A/B", ],
    "Integrability": ["integrat", "interoperab", "interface", "depend", "inject", "wrap", "bridg", "mediat", "adapter",
                      "contract", "protocol", "message", "synchroniz", "publish-subscribe", "pub-sub", "service bus",
                      "rout", ],
    "Modifiability": ["adapt", "evolve", "extend", "enhance", "flexible", "maintainab", "refactor", "rewrite", "config",
                      "parameteriz", "polymorphi", "inherit", "coupling", "cohesion", "layers", "sandbox", "variab",
                      "portab", "location independence", "plug-in", "plugin", "microkernel",
                      ],
    "Performance": ["perform", "latency", "throughput", "response time", "miss rate", "load", "scalab", "bottleneck",
                    "tune", "optimiz", "concurren", "multi-thread", "race condition", "queue", "cache", "throttle",
                    "load balanc", ],
    "Safety": ["safe", "unsafe", "hazard", "risk", "avoid", "detect", "remediat", "redund", "predict", "timeout",
               "sanity check", "abort", "interlock", "recover",
               ],
    "Security": ["secur", "attack", "intrusion", "denial-of-service", "confidential", "integrity", "authoriz", "access",
                 "firewall", "interlock", "encrypt", "validat", "sanitiz", "audit", "threat", "attack", "expose",
                 "checksum", "hash", "man-in-the-middle", "password", "certificate", "authenticat", "biometric",
                 "CAPTCHA", "inject", "cross-site scripting", "XSS", ],
    "Testability": ["test", "sandbox", "assert", "mock", "stub", "log", "resource monitor", "benchmark", ],
    "Usability": ["usab", "user-friendly", "user experience", "UX", "feedback", "undo", "pause", "resume",
                  "progress bar", "clear", "learn", "responsiv", "simpl", "guid", "intuit", ]}

quality_attributes_sample = {'sample': ['perf', "optimiz", "speed", "fast", "mode"]
                             }
