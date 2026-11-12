from pathlib import Path
import json
from collections import Counter

BENCHMARK = Path("benchmark/questions.json")

NEW_QUESTIONS = [
    # jess104 — Agriculture
    {
        "id": "JESS104-Q04",
        "question": "What are India's three main cropping seasons, and when are Rabi crops generally sown and harvested?",
        "type": "factual",
        "answerable": True,
        "expected_answer": "India has three cropping seasons: rabi, kharif and zaid. Rabi crops are sown in winter from October to December and harvested in summer from April to June.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess104.pdf",
                        "pages": [3],
                        "quote": "India has three cropping seasons — rabi, kharif and zaid. Rabi crops are sown in winter from October to December and harvested in summer from April to June."
                    }
                ],
                "okf_concept_ids": ["KC-JESS104-002"]
            }
        ]
    },
    {
        "id": "JESS104-Q05",
        "question": "Which states are major producers of Bajra, and what soil conditions favour its cultivation?",
        "type": "factual",
        "answerable": True,
        "expected_answer": "Bajra grows well on sandy soils and shallow black soil. Major producing states are Rajasthan, Uttar Pradesh, Maharashtra, Gujarat and Haryana.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess104.pdf",
                        "pages": [5],
                        "quote": "Bajra grows well on sandy soils and shallow black soil. Major Bajra producing States are Rajasthan, Uttar Pradesh, Maharashtra, Gujarat and Haryana."
                    }
                ],
                "okf_concept_ids": ["KC-JESS104-006"]
            }
        ]
    },
    {
        "id": "JESS104-Q06",
        "question": "How does commercial farming differ from subsistence farming, and why can the same crop be commercial in one region but subsistence in another?",
        "type": "conceptual",
        "answerable": True,
        "expected_answer": "Commercial farming uses modern inputs such as HYV seeds, chemical fertilisers, insecticides and pesticides to obtain higher productivity. The degree of commercialisation varies by region; rice is a commercial crop in Haryana and Punjab but a subsistence crop in Odisha.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess104.pdf",
                        "pages": [2],
                        "quote": "The main characteristic of this type of farming is the use of higher doses of modern inputs, e.g. high yielding variety (HYV) seeds, chemical fertilisers, insecticides and pesticides in order to obtain higher productivity. The degree of commercialisation of agriculture varies from one region to another. For example, rice is a commercial crop in Haryana and Punjab, but in Odisha, it is a subsistence crop."
                    }
                ],
                "okf_concept_ids": ["KC-JESS104-001"]
            }
        ]
    },
    {
        "id": "JESS104-Q07",
        "question": "How do plantation agriculture, processing industries, labour and transport networks depend on one another?",
        "type": "multi_hop",
        "answerable": True,
        "expected_answer": "Plantations grow a single crop over a large area using capital-intensive inputs and migrant labour; their produce is used as raw material by industries. The textbook describes tea as plantation agriculture and explains that tea processing is closely associated with the producing gardens, while transport and communication connect production areas with processing and markets.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess104.pdf",
                        "pages": [2],
                        "quote": "Plantation is also a type of commercial farming. In this type of farming, a single crop is grown on a large area. The plantation has an interface of agriculture and industry. Plantations cover large tracts of land, using capital intensive inputs, with the help of migrant labourers. All the produce is used as raw material in respective industries. In India, tea, coffee, rubber, sugarcane, banana, etc., are important plantation crops. Since the production is mainly for market, a well-developed network of transport and communication connecting the plantation areas, processing industries and markets plays an important role in the development of plantations."
                    }
                ],
                "okf_concept_ids": ["KC-JESS104-001"]
            },
            {
                "id": "E2",
                "relevance": 2,
                "pdf_anchors": [
                    {
                        "document": "jess104.pdf",
                        "pages": [8],
                        "quote": "Tea is a labour-intensive industry. It requires abundant, cheap and skilled labour. Tea is processed within the tea garden to restore its freshness."
                    }
                ],
                "okf_concept_ids": ["KC-JESS104-011"]
            }
        ]
    },
    {
        "id": "JESS104-Q08",
        "question": "How do land fragmentation and limited alternative livelihoods contribute to pressure on agricultural land, and how do land-reform movements attempt to address land inequality?",
        "type": "multi_hop",
        "answerable": True,
        "expected_answer": "Inheritance fragments holdings into uneconomical sizes. With few alternative livelihoods, farmers seek maximum output from limited land, increasing pressure. Land reforms sought to consolidate fragmented holdings; Bhoodan and Gramdan encouraged voluntary land donations or village-level redistribution for landless people.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess104.pdf",
                        "pages": [2],
                        "quote": "Though the ‘right of inheritance’ leading to the division of land among successive generations has rendered land-holding size uneconomical, the farmers continue to take maximum output from the limited land in the absence of alternative source of livelihood. Thus, there is enormous pressure on agricultural land."
                    }
                ],
                "okf_concept_ids": ["KC-JESS104-001"]
            },
            {
                "id": "E2",
                "relevance": 2,
                "pdf_anchors": [
                    {
                        "document": "jess104.pdf",
                        "pages": [9],
                        "quote": "‘Land reform’ was the main focus of our First Five Year Plan. The right of inheritance had already led to fragmentation of land holdings necessitating consolidation of holdings."
                    }
                ],
                "okf_concept_ids": ["KC-JESS104-016"]
            },
            {
                "id": "E3",
                "relevance": 2,
                "pdf_anchors": [
                    {
                        "document": "jess104.pdf",
                        "pages": [10],
                        "quote": "Suddenly, Shri Ram Chandra Reddy stood up and offered 80 acres of land to be distributed among 80 land-less villagers. This act was known as ‘Bhoodan’. Later he travelled and introduced his ideas widely all over India. Some zamindars, owners of many villages offered to distribute some villages among the landless. It was known as Gramdan."
                    }
                ],
                "okf_concept_ids": ["KC-JESS104-019"]
            }
        ]
    },

    # jess105 — Minerals and Energy Resources
    {
        "id": "JESS105-Q04",
        "question": "What are the conventional and non-conventional sources of energy described in the textbook?",
        "type": "factual",
        "answerable": True,
        "expected_answer": "Conventional sources include firewood, cattle dung cake, coal, petroleum, natural gas and electricity (hydel and thermal). Non-conventional sources include solar, wind, tidal, geothermal, biogas and atomic energy.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess105.pdf",
                        "pages": [9],
                        "quote": "Conventional sources include: firewood, cattle dung cake, coal, petroleum, natural gas and electricity (both hydel and thermal). Non-conventional sources include solar, wind, tidal, geothermal, biogas and atomic energy."
                    }
                ],
                "okf_concept_ids": ["KC-JESS105-011"]
            }
        ]
    },
    {
        "id": "JESS105-Q05",
        "question": "How do the geological formations in which minerals occur affect their extraction and cost?",
        "type": "conceptual",
        "answerable": True,
        "expected_answer": "Minerals occur in different geological formations, including veins and lodes in igneous and metamorphic rocks, beds or layers in sedimentary rocks, and residual deposits. The type of formation determines the relative ease of mining and the cost of extraction.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess105.pdf",
                        "pages": [2],
                        "quote": "The type of formation or structure in which they are found determines the relative ease with which mineral ores may be mined. This also determines the cost of extraction."
                    }
                ],
                "okf_concept_ids": ["KC-JESS105-001"]
            }
        ]
    },
    {
        "id": "JESS105-Q06",
        "question": "Why does mining create both occupational health risks and environmental degradation?",
        "type": "conceptual",
        "answerable": True,
        "expected_answer": "Dust and noxious fumes expose miners to pulmonary diseases, while collapsing roofs, inundation and fires threaten their safety. Mining contaminates water sources, and dumping waste and slurry degrades land and soil and increases stream and river pollution.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess105.pdf",
                        "pages": [7],
                        "quote": "The dust and noxious fumes inhaled by miners make them vulnerable to pulmonary diseases. The risk of collapsing mine roofs, inundation and fires in coalmines are a constant threat to miners."
                    }
                ],
                "okf_concept_ids": ["KC-JESS105-009"]
            },
            {
                "id": "E2",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess105.pdf",
                        "pages": [7],
                        "quote": "The water sources in the region get contaminated due to mining. Dumping of waste and slurry leads to degradation of land, soil, and increase in stream and river pollution."
                    }
                ],
                "okf_concept_ids": ["KC-JESS105-009"]
            }
        ]
    },
    {
        "id": "JESS105-Q07",
        "question": "How do the distribution of iron ore across India's major belts and the transport of extracted ore influence the mineral economy?",
        "type": "multi_hop",
        "answerable": True,
        "expected_answer": "India's iron ore is distributed across major belts including Odisha-Jharkhand, Durg-Bastar-Chandrapur and Ballari-Chitradurga-Chikkamagaluru-Tumakuru. The textbook gives examples of ore being exported through Visakhapatnam and transported as slurry by pipeline from Kudremukh to a port near Mangaluru, showing the role of transport links in moving mineral output to markets.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess105.pdf",
                        "pages": [4],
                        "quote": "The major iron ore belts in India are: Odisha-Jharkhand belt; Durg-Bastar-Chandrapur belt; Ballari-Chitradurga-Chikkamagaluru-Tumakuru belt in Karnataka."
                    }
                ],
                "okf_concept_ids": ["KC-JESS105-003"]
            },
            {
                "id": "E2",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess105.pdf",
                        "pages": [4],
                        "quote": "Iron ore from these mines is exported to Japan and South Korea via Vishakhapatnam port."
                    }
                ],
                "okf_concept_ids": ["KC-JESS105-003"]
            },
            {
                "id": "E3",
                "relevance": 2,
                "pdf_anchors": [
                    {
                        "document": "jess105.pdf",
                        "pages": [5],
                        "quote": "The ore is transported as slurry through a pipeline to a port near Mangaluru."
                    }
                ],
                "okf_concept_ids": ["KC-JESS105-003"]
            }
        ]
    },
    {
        "id": "JESS105-Q08",
        "question": "How do the uses of natural gas across power, industry, transport and households explain the importance of pipeline infrastructure?",
        "type": "multi_hop",
        "answerable": True,
        "expected_answer": "Natural gas is used to generate electricity, heat industries, provide raw material for chemical, petrochemical and fertiliser industries, fuel transport (CNG) and supply household cooking gas (PNG). The HVJ pipeline linked gas fields with fertiliser, power and industrial complexes, while the wider pipeline grid links gas sources with consuming markets.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess105.pdf",
                        "pages": [11],
                        "quote": "It is used as fuel in power sector to generate electricity, for heating purpose in industries, as raw material in chemical, petrochemical and fertilizer industries, as transport fuel and as cooking fuel."
                    }
                ],
                "okf_concept_ids": ["KC-JESS105-012"]
            },
            {
                "id": "E2",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess105.pdf",
                        "pages": [11, 12],
                        "quote": "The first 1,700 km long Hazira-Vijaipur-Jagdishpur (HVJ) cross country gas pipeline, constructed by GAIL (India), linked Mumbai High and Bassein gas fields with various fertilizer, power and industrial complexes in western and northen India."
                    }
                ],
                "okf_concept_ids": ["KC-JESS105-012"]
            }
        ]
    },

    # jess106 — Manufacturing Industries
    {
        "id": "JESS106-Q04",
        "question": "What are the principal raw materials used in iron and steel production, and in what approximate ratio are they required?",
        "type": "factual",
        "answerable": True,
        "expected_answer": "Iron ore, coking coal and limestone are required in an approximate ratio of 4:2:1. Manganese is also required to harden steel.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess106.pdf",
                        "pages": [5],
                        "quote": "Iron ore, coking coal and lime stone are required in a ratio of approximately 4:2:1. Manganese is also required to harden the steel."
                    }
                ],
                "okf_concept_ids": ["KC-JESS106-004", "KC-JESS106-005"]
            }
        ]
    },
    {
        "id": "JESS106-Q05",
        "question": "Why is the Chhotanagpur Plateau region favourable for the concentration of iron and steel industries?",
        "type": "conceptual",
        "answerable": True,
        "expected_answer": "The region offers low-cost iron ore, high-grade raw materials nearby, cheap labour and growth potential in the home market.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess106.pdf",
                        "pages": [7],
                        "quote": "Chhotanagpur plateau region has the maximum concentration of iron and steel industries. It is largely, because of the relative advantages this region has for the development of this industry. These include, low cost of iron ore, high grade raw materials in proximity, cheap labour and vast growth potential in the home market."
                    }
                ],
                "okf_concept_ids": ["KC-JESS106-006"]
            }
        ]
    },
    {
        "id": "JESS106-Q06",
        "question": "How do industrial activities cause air, water (including thermal), land and noise pollution, and what measures can industries take to reduce these effects?",
        "type": "multi_hop",
        "answerable": True,
        "expected_answer": "The textbook identifies air, water, land and noise as four pollution categories; thermal pollution is described as a form of water pollution caused by hot water discharged into rivers. Control measures include treating industrial effluents, reducing particulate matter in air, and using appropriate treatment and pollution-control equipment.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess106.pdf",
                        "pages": [9],
                        "quote": "Industries are responsible for four types of pollution: (a) Air (b) Water (c) Land (d) Noise."
                    }
                ],
                "okf_concept_ids": ["KC-JESS106-013"]
            },
            {
                "id": "E2",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess106.pdf",
                        "pages": [9],
                        "quote": "Thermal pollution of water occurs when hot water from factories is drained into rivers."
                    }
                ],
                "okf_concept_ids": ["KC-JESS106-013"]
            },
            {
                "id": "E3",
                "relevance": 2,
                "pdf_anchors": [
                    {
                        "document": "jess106.pdf",
                        "pages": [11],
                        "quote": "(i) minimising use water for processing by reusing and recycling it in two or more successive stages (ii) harvesting of rainwater to meet water requirements (iii) treating hot water and effluents before releasing them in rivers and ponds."
                    }
                ],
                "okf_concept_ids": ["KC-JESS106-013", "KC-JESS106-014"]
            }
        ]
    },
    {
        "id": "JESS106-Q07",
        "question": "Why is iron and steel considered a basic industry, and how does its output support other manufacturing sectors?",
        "type": "multi_hop",
        "answerable": True,
        "expected_answer": "Iron and steel is a basic industry because heavy, medium and light industries depend on it for machinery. Steel is used in engineering goods, construction materials, defence, medical, telephonic and scientific equipment, and consumer goods.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess106.pdf",
                        "pages": [5],
                        "quote": "The iron and steel industry is the basic industry since all the other industries — heavy, medium and light, depend on it for their machinery."
                    }
                ],
                "okf_concept_ids": ["KC-JESS106-004"]
            },
            {
                "id": "E2",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess106.pdf",
                        "pages": [5],
                        "quote": "Steel is needed to manufacture a variety of engineering goods, construction material, defence, medical, telephonic, scientific equipment and a variety of consumer goods."
                    }
                ],
                "okf_concept_ids": ["KC-JESS106-004", "KC-JESS106-005"]
            }
        ]
    },

    # jess107 — Lifelines of National Economy
    {
        "id": "JESS107-Q04",
        "question": "What are the five National Waterways listed in the textbook, and which routes do they cover?",
        "type": "factual",
        "answerable": True,
        "expected_answer": "NW-1: Ganga between Prayagraj and Haldia; NW-2: Brahmaputra between Sadiya and Dhubri; NW-3: West-Coast Canal in Kerala; NW-4: specified stretches of the Godavari and Krishna; NW-5: specified stretches of the Brahmani and Matai rivers, Mahanadi and Brahmani delta channels, and the East Coast Canal.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess107.pdf",
                        "pages": [7],
                        "quote": "The Ganga river between Prayagraj and Haldia (1620 km)-N.W. No.1 The Brahmaputra river between Sadiya and Dhubri (891 km)-N.W. No.2 West-Coast Canal in Kerala (205 km)-N.W. No.3 Specified stretches of Godavari and Krishna rivers (1078 km)-N.W. No.4 Specified stretches of river Brahmani along with Matai river, delta channels of Mahanadi and Brahmani rivers and East Coast Canal (588 km)-N.W. No.5"
                    }
                ],
                "okf_concept_ids": ["KC-JESS107-009"]
            }
        ]
    },
    {
        "id": "JESS107-Q05",
        "question": "What are the main reasons roadways have an advantage over railways in certain transport situations?",
        "type": "factual",
        "answerable": True,
        "expected_answer": "Roads are cheaper to construct and maintain, can cross dissected and undulating terrain and steep gradients, are economical for a few people or smaller quantities of goods over short distances, provide door-to-door service, and act as feeders to other transport modes.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess107.pdf",
                        "pages": [2],
                        "quote": "construction cost of roads is much lower than that of railway lines"
                    }
                ],
                "okf_concept_ids": ["KC-JESS107-001", "KC-JESS107-002"]
            },
            {
                "id": "E2",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess107.pdf",
                        "pages": [2],
                        "quote": "road transport is economical in transportation of few persons and relatively smaller amount of goods over short distances"
                    }
                ],
                "okf_concept_ids": ["KC-JESS107-002"]
            },
            {
                "id": "E3",
                "relevance": 2,
                "pdf_anchors": [
                    {
                        "document": "jess107.pdf",
                        "pages": [2],
                        "quote": "it also provides door-to-door service, thus the cost of loading and unloading is much lower"
                    }
                ],
                "okf_concept_ids": ["KC-JESS107-002"]
            }
        ]
    },
    {
        "id": "JESS107-Q06",
        "question": "How do transport, communication and trade complement one another and support national economic development?",
        "type": "conceptual",
        "answerable": True,
        "expected_answer": "The textbook describes transport, communication and trade as complementary. Transport moves people and goods, communication supports the exchange of information, and trade connects producers and markets; together, these systems act as lifelines of the national economy.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess107.pdf",
                        "pages": [1, 2],
                        "quote": "Transport, communication, and trade are complementary to each other. India is well-linked with the rest of the world despite its vast size, diversity, and linguistic and socio-cultural plurality. Modern means of transport and communication serve as lifelines of our nation and its modern economy."
                    }
                ],
                "okf_concept_ids": ["KC-JESS107-001"]
            },
            {
                "id": "E2",
                "relevance": 2,
                "pdf_anchors": [
                    {
                        "document": "jess107.pdf",
                        "pages": [10],
                        "quote": "Export and import are the components of trade. The balance of trade of a country is the difference between its export and import."
                    }
                ],
                "okf_concept_ids": ["KC-JESS107-013"]
            }
        ]
    },
    {
        "id": "JESS107-Q07",
        "question": "How do pipelines compare with other modes of transport in operating costs, delays and suitability for transporting materials?",
        "type": "conceptual",
        "answerable": True,
        "expected_answer": "Pipelines can transport liquids, gases and some slurry materials. Although the initial cost of laying pipelines is high, subsequent running costs are minimal, and pipelines avoid trans-shipment losses and delays.",
        "evidence_units": [
            {
                "id": "E1",
                "relevance": 3,
                "pdf_anchors": [
                    {
                        "document": "jess107.pdf",
                        "pages": [5],
                        "quote": "Initial cost of laying pipelines is high but subsequent running costs are minimal. It rules out trans-shipment losses or delays."
                    }
                ],
                "okf_concept_ids": ["KC-JESS107-006", "KC-JESS107-008"]
            }
        ]
    }
]


def main():
    if not BENCHMARK.exists():
        raise SystemExit(
            f"Cannot find {BENCHMARK}. Run this script from evaluation-framework."
        )

    existing = json.loads(BENCHMARK.read_text(encoding="utf-8"))

    if not isinstance(existing, list):
        raise SystemExit("questions.json must contain a JSON array.")

    new_ids = [question["id"] for question in NEW_QUESTIONS]

    if len(new_ids) != len(set(new_ids)):
        raise SystemExit("The append payload contains duplicate IDs.")

    existing_ids = {question["id"] for question in existing}
    duplicates = existing_ids.intersection(new_ids)

    if duplicates:
        raise SystemExit(
            f"Refusing to append duplicate IDs: {sorted(duplicates)}"
        )

    if len(existing) != 9:
        raise SystemExit(
            f"Expected the approved 9-question benchmark before append; "
            f"found {len(existing)}. No changes made."
        )

    backup = BENCHMARK.with_suffix(".json.bak")
    backup.write_text(
        BENCHMARK.read_text(encoding="utf-8"),
        encoding="utf-8"
    )

    combined = existing + NEW_QUESTIONS

    BENCHMARK.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    question_types = Counter(
        question["type"] for question in combined
    )

    print(f"Appended: {len(NEW_QUESTIONS)}")
    print(f"Total questions: {len(combined)}")
    print(f"Question types: {dict(sorted(question_types.items()))}")
    print(f"Backup: {backup}")
    print("IDs appended:")

    for question_id in new_ids:
        print(f"  {question_id}")


if __name__ == "__main__":
    main()