# Transmission Portfolios and Operations for 2035 Scenarios

National Transmission Planning Study

Chapter 3:

Transmission Portfolios and
Operations for 2035 Scenarios

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

This report is being disseminated by the Department of Energy. As such, this document
was prepared in compliance with Section 515 of the Treasury and General Government
Appropriations Act for Fiscal Year 2001 (Public Law 106-554) and information quality
guidelines issued by the Department of Energy.
Suggested citation
U.S. Department of Energy, Grid Deployment Office. 2024. The National
Transmission Planning Study. Washington, D.C.: U.S. Department of Energy.
https://www.energy.gov/gdo/national-transmission-planning-study.

National Transmission Planning Study

i

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Context
The National Transmission Planning Study (NTP Study) is presented as a collection of
six chapters and an executive summary, each of which is listed next. The NTP Study
was led by the U.S. Department of Energy's Grid Deployment Office, in partnership with
the National Renewable Energy Laboratory and Pacific Northwest National Laboratory.
•

The Executive Summary describes the high-level findings from across all six
chapters and next steps for how to build on the analysis.

•

Chapter 1: Introduction provides background and context about the technical
design of the study and modeling framework, introduces the scenario framework,
and acknowledges those who contributed to the study.

•

Chapter 2: Long-Term U.S. Transmission Planning Scenarios discusses the
methods for capacity expansion and resource adequacy, key findings from the
scenario analysis and economic analysis, and High Opportunity Transmission
interface analysis.

•

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios (this
chapter) summarizes the methods for translating zonal scenarios to nodalnetwork-level models, network transmission plans for a subset of the scenarios,
and key findings from transmission planning and production cost modeling for the
contiguous United States.

•

Chapter 4: AC Power Flow Analysis for 2035 Scenarios identifies the
methods for translating from zonal and nodal production cost models to
alternating current (AC) power flow models and describes contingency analysis
for a subset of scenarios.

•

Chapter 5: Stress Analysis for 2035 Scenarios outlines how the future
transmission expansions perform under stress tests.

•

Chapter 6: Conclusions describes the high-level findings and study limitations
across the six chapters.

As of publication, there are three additional reports under the NTP Study umbrella that
explore related topics, each of which is listed next. 1 For more information on the NTP
Study, visit https://www.energy.gov/gdo/national-transmission-planning-study.
•

Interregional Renewable Energy Zones connects the NTP Study scenarios to
ground-level regulatory and financial decision making—specifically focusing on
the potential of interregional renewable energy zones.

In addition to these three reports, the DOE and laboratories are exploring future analyses of the
challenges within the existing interregional planning landscape and potential regulatory and industry
solutions.
1

National Transmission Planning Study

ii

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

•

Barriers and Opportunities To Realize the System Value of Interregional
Transmission examines issues that prevent existing transmission facilities from
delivering maximum potential value and offers a suite of options that power
system stakeholders can pursue to overcome those challenges between
nonmarket or a mix of market and nonmarket areas and between market areas.

•

Western Interconnection Baseline Study uses production cost modeling to
compare a 2030 industry planning case of the Western Interconnection to a high
renewables case with additional planned future transmission projects based on
best available data.

National Transmission Planning Study

iii

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

List of Acronyms
ADS

Anchor Dataset

AC

alternating current

AFUDC

allowance for funds used during construction

ATB

Annual Technology Baseline

EI

Eastern Interconnection

BA

balancing authority

BASN

Basin (WECC Region)

B2B

back-to-back

BESS

battery energy storage systems

BLM

Bureau of Land Management

CAISO

California Independent System Operator

CALN

Northern California (WECC region)

CALS

Southern California (WECC region)

CC

combined cycle

CCS

carbon capture and storage

CEM

Capacity Expansion Model

CEMS

continuous emission monitoring system

CONUS

contiguous United States

DC

direct current

DOE

U.S. Department of Energy

DSW

Desert Southwest (WECC region)

EER

Evolved Energy Research

EI

Eastern Interconnection

EIA

U.S. Energy Information Administration

National Transmission Planning Study

iv

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

EIPC

Eastern Interconnection Planning Collaborative

EHV

extra high voltage (>500 kV)

ENTSO-E

European Network of Transmission System Operators for Electricity

EPE

Empresa de Pesquisa Energética

ERAG

Eastern Interconnection Reliability Assessment Group

ERCOT

Electric Reliability Council of Texas

FRCC

Florida Reliability Coordinating Council

GT

gas turbine

GW

gigawatt

HV

high voltage (230 kV ≥ V > 500 kV)

HVAC

high-voltage alternating current

HVDC

high-voltage direct current

IBR

inverter-based resource

ISO

independent system operator

ISONE

Independent System Operator of New England

ITC

investment tax credit

km

kilometer

kV

kilovolt

LCC

line-commutated converter

LCOE

levelized cost of energy

LPF

load participation factor

MISO

Midcontinent Independent System Operator

MMWG

Multiregion Modeling Working Group

MT

multiterminal

MVA

megavolt ampere

National Transmission Planning Study

v

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

MW

megawatt

NARIS

North American Renewable Integration Study

NERC

North American Electric Reliability Corporation

NREL

National Renewable Energy Laboratory

NTP Study

National Transmission Planning Study

NWUS-E

Northwest U.S. West (WECC region)

NWUS-W

Northwest U.S. East (WECC Region)

NYISO

New York Independent System Operator

PCM

production cost model

POI

point of interconnection

PPA

power purchase agreement

PTC

production tax credit

PV

photovoltaic

R&D

research and development

ReEDS

Regional Energy Deployment System (model)

reV

Renewable Energy Potential (model)

ROCK

Rocky Mountain (WECC Region)

ROW

right of way

SERTP

Southeastern Regional Transmission Planning

SPP

Southwest Power Pool

TI

Texas Interconnection

TRC

technical review committee

TWh

terawatt-hour

TW-miles

terawatt-miles

VSC

voltage source converter

National Transmission Planning Study

vi

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

VRE

variable renewable energy

WECC

Western Electricity Coordinating Council

WG

Working Group

WI

Western Interconnection

Z2N

zonal-to-nodal

National Transmission Planning Study

vii

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Chapter 3 Overview
The National Transmission Planning Study (NTP Study) evaluates the role and value of
transmission for the contiguous United States. The development and verification of
transmission portfolios is a critical aspect of the analysis that allows transformative
scenarios to be tested against network and operational constraints. This chapter
presents the methods for developing the transmission portfolios, the resulting portfolios
for the contiguous United States, and alternative portfolios for the Western
Interconnection used for focused studies in Chapters 4 and 5 as well as additional
insights about how these transmission portfolios are used in hourly operations for a
future model year of 2035. The chapter also describes an evaluation of the benefits and
costs of transmission for the alternative Western Interconnection portfolios.
The transmission portfolio analysis adopts a subset of future scenarios from the
capacity expansion modeling (Regional Energy Deployment System [ReEDS]
scenarios; Chapter 2) to translate into nodal production cost and linearized direct
current (DC) power flow models. The process for translating from zonal ReEDS
scenarios to nodal models required the integration of industry planning power flow
cases for each interconnection. The generation and storage capacities from the ReEDS
scenarios were strictly adopted. However, given the additional constraints of planning
transmission in a detailed network model, the ReEDS transmission capacities and zonal
connections were guidelines in the transmission planning stage. The resulting
transmission portfolios reflect the general trends from the ReEDS scenarios. In addition,
given the broad possibilities of a transmission portfolio for the entire contiguous United
States, the focus of the transmission buildout started with interregional needs, followed
by other high-voltage intraregional transfers and local needs.
The three ReEDS scenarios adopted for the zonal-to-nodal translation are shown in
Table I. These scenarios represent central assumptions from the full set of 96 ReEDS
scenarios, representing the Mid-Demand, 90% by 2035 decarbonization, and central
technology costs for the three transmission frameworks.

National Transmission Planning Study

viii

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Table I. Summary of Scenarios for Zonal-to-Nodal Translation

Dimension

Limited

AC

MT-HVDC

Transmission framework1

AC expansion within
transmission planning
regions

AC expansion within
interconnects

HVDC expansion across
interconnects
(+AC within transmission
planning regions)

Model year

Annual electricity demand

CO2 emissions target
1

2035
Mid Demand1
CONUS: 5620 TWh (916 GW)
Western Interconnection: 1097 TWh (186 GW)
ERCOT: 509 TWh (93 GW)
Eastern Interconnection: 4014 TWh (665 GW)
CONUS: 90% reduction by 2035
(relative to 2005)

See Chapter 2 for further details.

CO2 = carbon dioxide; AC = alternating current; TWh = terawatt-hour; GW = gigawatt; HVDC = high-voltage direct current

The resulting transmission portfolios developed for the nodal analysis represent one of
many possible network expansions that meet the needs of the zonal scenarios. Figures
I through III show the three portfolios developed for the contiguous United States. The
vast differences between transmission portfolios that all reach a 90% reduction in
emissions by 2035 demonstrate that transmission can enable multiple pathways to
decarbonization. In addition, all transmission portfolios demonstrate intraregional
networks as an important component of expansion and high-voltage networks to collect
renewable energy in futures with high amounts of wind and solar.

National Transmission Planning Study

ix

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure I. Transmission portfolio for the Limited scenario for model year 2035

Figure II. Transmission portfolio for the AC scenario for model year 2035
AC = alternating current

National Transmission Planning Study

x

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure III. Transmission portfolio for the MT-HVDC scenario for the model year 2035
MT = multiterminal; HVDC = high-voltage direct current

The production cost modeling of the scenarios demonstrates the addition of substantial
amounts of interregional transmission provides significant opportunities and challenges
for grid operators. For example, most regions rely on imports or exports in all three
scenarios, but across almost all regions, the alternating current (AC) and
multiterminal (MT) high-voltage direct current (HVDC) transmission scenarios lead to
more overall energy exchange. More specifically, 19% of the total energy consumed in
the Limited scenario flows over interregional transmission lines whereas that number
increases to 28% in AC and 30% in the MT-HVDC scenario.
The analysis of the Western Interconnection examines how flows around the West
change in response to high levels of renewable energy and transmission. In the AC and
MT-HVDC scenarios, there is more variation on the interregional transmission lines,
including larger swings diurnally. The patterns of flow are impacted by the generation
from solar and the role storage plays in meeting peak demand. On average, the Limited
scenario relies on more storage generation during the peak than the AC and MT-HVDC
scenarios. In addition, the MT-HVDC scenario also imports across the HVDC lines
connecting the Eastern Interconnection to balance the availability of generation in the
West, which reduces the need for peaking storage.
The methods developed for the transmission portfolio analysis are novel in that they
could be applied on a large geographic scale and include both production cost modeling
and rapid DC-power-flow-informed transmission expansions. The incorporation of
capacity expansion modeling data was also innovative. This is not the first example of
National Transmission Planning Study

xi

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

closely linking capacity expansion modeling to production cost and power flow models
in industry or the research community. However, several advancements were made to
realistically capture the various network points of interconnection for large amounts of
wind and solar, build out the local collector networks if necessary, and validate the
interregional portfolios that might arise from scenarios that reach 90% reduction in
emissions by 2035.

National Transmission Planning Study

xii

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Table of Contents
1

Introduction .............................................................................................................. 1

2

Methodology............................................................................................................. 3
2.1

3

Zonal-to-Nodal Translation ................................................................................ 6

2.1.1

Geographic extent and nodal datasets ...................................................... 7

2.1.2

Modeling domains and tools ...................................................................... 9

2.2

Disaggregation ................................................................................................ 11

2.3

Transmission Expansion ................................................................................. 13

2.4

Transmission Planning Feedback to Capacity Expansion............................... 19

2.5

Scenarios for Nodal Transmission Plans and Production Cost Modeling........ 21

2.6

Economic Analysis of the Western Interconnection (earlier scenario results) . 23

2.6.1

Transmission capital cost methodology ................................................... 23

2.6.2

Generation capital cost methodology ....................................................... 23

2.6.3

Operating cost economic metrics methodology ....................................... 24

2.6.4

Net annualized avoided cost methodology .............................................. 24

2.6.5

Benefit disaggregation and calculating the annualized net present value 24

Contiguous U.S. Results for Nodal Scenarios ........................................................ 28
3.1
A Range of Transmission Topologies Enables High Levels of
Decarbonization ......................................................................................................... 28
3.2

Translating Zonal Scenarios to Nodal Network Scenarios .............................. 31

3.2.1

Scale and dispersion of new resources is unprecedented ....................... 31

3.2.2
Intraregional transmission needs are substantial, especially when
interregional options are not available .................................................................... 37
3.2.3
Achieving high levels of interregional power exchanges using AC
transmission technologies requires long-distance, high-capacity HV corridors
combined with intraregional reinforcements ........................................................... 39
3.2.4
HVDC transmission buildout represents a paradigm shift and includes the
adoption of technologies currently not widespread in the United States ................ 42
3.3

Operations of Highly Decarbonized Power Systems....................................... 46

3.3.1
Interregional transmission is highly used to move renewable power to load
centers but also to balance resources across regions ........................................... 46
3.3.2
Diurnal and seasonal variability may require increased flexibility as well as
interregional coordination to minimize curtailment ................................................. 49

National Transmission Planning Study

xiii

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

3.3.3
Regions with high amounts of VRE relative to demand become major
exporters and often exhibit very low amounts of synchronous generation ............. 56
4

Western Interconnection Results for Downstream Modeling .................................. 59
4.1

Translating Zonal Scenarios to Nodal Network Scenarios .............................. 59

4.1.1
Increased transmission expansion in the West via long-distance highcapacity lines could enable lower overall generation capacity investment by
connecting the resources far from load centers ..................................................... 59
4.2

Operations of Highly Decarbonized Power Systems....................................... 63

4.2.1
Operational flexibility is achieved by a changing generation mix that
correlates with the amount of interregional transmission capacity ......................... 63
4.2.2
Transmission paths connecting diverse VRE resources will experience
more bidirectional flow and diurnal patterns ........................................................... 66
4.2.3
HVDC links between the Western and Eastern Interconnections are highly
used and exhibit geographically dependent bidirectional flows .............................. 67
4.3
Economic Analysis Indicates Benefits From More Interregional Transmission in
the Considered Scenarios ......................................................................................... 68
4.3.1
Increased transmission capital cost expenditures in the studied
interregional scenarios coincide with lower generation capital costs...................... 69
4.3.2
Operating costs decrease with increased interregional transmission,
resulting in greater net annualized benefits ............................................................ 70
5

Conclusions............................................................................................................ 76
5.1

Opportunities for Further Research................................................................. 77

References .................................................................................................................... 78
Appendix A. Methodology ............................................................................................. 83
Appendix B. Scenario Details ........................................................................................ 94
Appendix C. Tools ....................................................................................................... 121
Appendix D. Economic Methodology........................................................................... 123

National Transmission Planning Study

xiv

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

List of Figures

Figure I. Transmission portfolio for the Limited scenario for model year 2035 ................ x
Figure II. Transmission portfolio for the AC scenario for model year 2035 ...................... x
Figure III. Transmission portfolio for the MT-HVDC scenario for the model year 2035 ....xi
Figure 1. Conceptual outline of zonal-to-nodal translation .............................................. 4
Figure 2. Overview of zonal-to-nodal translation approach ............................................. 7
Figure 3. Map of the three interconnections of the bulk U.S. power system.................... 8
Figure 4. Overview of the zonal results flowing into the nodal model ............................ 11
Figure 5. Illustration of VRE disaggregation process as part of Z2N translation ........... 13
Figure 6. Stages of transmission expansion planning ................................................... 14
Figure 7. Prioritization of transmission expansion planning approach ........................... 17
Figure 8. Illustration of single contingency definition for Stage 3 of zonal-to-nodal
process .................................................................................................................... 18
Figure 9. Illustration of mitigation of causes and consequences during contingency
analysis .................................................................................................................... 19
Figure 10. Overview of transmission expansion feedback between zonal and nodal
modeling domains .................................................................................................... 20
Figure 11. Interregional transfer capacity from ReEDS zonal scenarios used for nodal
Z2N scenarios .......................................................................................................... 22
Figure 12. Generation and storage capacity for final nodal scenarios ........................... 32
Figure 13. Nodal POIs, sized by capacity, for all generation types for the Limited
scenario ................................................................................................................... 34
Figure 14. Nodal POIs, sized by capacity, for all generation types for the AC scenario 35
Figure 15. Nodal POIs, sized by capacity, for all generation types for the MT-HVDC
scenario ................................................................................................................... 36
Figure 16. Nodal transmission expansion solution for the Limited scenario for model
year 2035 ................................................................................................................. 38
Figure 17. Nodal transmission expansion solution for the AC Scenario for the model
year 2035 ................................................................................................................. 41
Figure 18. Transmission portfolio solution for MT-HVDC scenario for the model year
2035 ......................................................................................................................... 45
Figure 19. Flow duration curve between MISO-Central and the Southeast................... 46
Figure 20. Flow duration curves (a) and distribution of flows (b), (c), (d) between FRCC
and the Southeast .................................................................................................... 48
Figure 21. Dispatch stack for peak demand period in FRCC and Southeast for (a) AC
and (b) MT-HVDC .................................................................................................... 49
Figure 22. VRE production duration curve (normalized to demand) for the Eastern
Interconnection ........................................................................................................ 50
Figure 23. Monthly generation (per interconnection) for the AC scenario...................... 51
Figure 24. Monthly generation (CONUS) for (a) MT-HVDC scenario and (b) within SPP
and (c) MISO ........................................................................................................... 52
Figure 25. Seasonal dispatch stacks (per interconnect) for the Limited scenario .......... 54

National Transmission Planning Study

xv

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 26. Seasonal dispatch stacks (per interconnect) for the AC scenario ................ 55
Figure 27. Seasonal dispatch stacks (continental United States) for the MT-HVDC
scenario ................................................................................................................... 56
Figure 28. Dispatch stacks of peak demand for SPP (top) and MISO (bottom) for the AC
scenario ................................................................................................................... 57
Figure 29. Ratio of Net interchange to for the 11 regions all hours of the year (2035) for
Limited, AC, and MT-HVDC nodal scenarios ........................................................... 58
Figure 30. Net installed capacity after disaggregation: (a) Western Interconnection-wide;
(b) by subregion ....................................................................................................... 61
Figure 31. Transmission expansion results on the Western Interconnection footprint ... 62
Figure 32. Annual generation mix comparison for (a) Western Interconnection and (b) by
subregion ................................................................................................................. 64
Figure 33. Average weekday in the third quarter of the model year for (a) Limited, (b)
AC, and (c) MT-HVDC ............................................................................................. 65
Figure 34. Average weekday storage dispatch for the Western Interconnection ........... 65
Figure 35. Flows between Basin and Southern California and between Desert
Southwest and Southern California.......................................................................... 66
Figure 36. Interface between Basin and Pacific Northwest ........................................... 67
Figure 37. Flow duration curve (left) and average weekday across four Western-Eastern
Interconnection seam corridors ................................................................................ 68
Figure A-1. Illustration of zonal-to nodal (Z2N) disaggregation of demand (a) CONUS
and (b) zoom to Colorado ........................................................................................ 86
Figure A-2. Illustration of nodal transmission network constraint formulations .............. 87
Figure A-3. Branch loading in the MT-HVDC scenario .................................................. 91
Figure A-4. Coupled MT-HVDC design concept and rationale ...................................... 93
Figure A-5. HVDC buildout in the MT-HVDC scenario................................................... 93
Figure B-1. Subtransmission planning regions (derived and aggregated from Regional
Energy Deployment System [ReEDS] regions) ........................................................ 96
Figure B-2. Subregion definitions used in Section 4 (earlier ReEDS scenario results) . 97
Figure B-3. Interregional transfer capacity from zonal ReEDS scenarios.................... 102
Figure B-4. Summary of installed capacity by transmission planning region,
interconnection and contiguous U.S. (CONUS)-wide............................................. 103
Figure B-5. Summary of nodal transmission expansion portfolios ............................... 104
Figure B-6. Curtailment comparison between scenarios in the Western
Interconnection ...................................................................................................... 114
Figure B-7. Relationship between solar and wind curtailment in the Western
Interconnection ...................................................................................................... 115
Figure B-8. Nodal disaggregated installed capacity for Western Interconnection and
Eastern Interconnection ......................................................................................... 116
Figure B-9. Transmission expansion for MT-HVDC scenario for Western and Eastern
Interconnection ...................................................................................................... 116
Figure B-10. Generation dispatch for combined Western Interconnection and Eastern
Interconnection ...................................................................................................... 117
National Transmission Planning Study

xvi

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure B-11. Curtailment for combined Western Interconnection and Eastern
Interconnection footprint ........................................................................................ 118
Figure B-12. Use of HVDC interregional interfaces in the MT-HVDC scenario (Western
Interconnection) ..................................................................................................... 119
Figure B-13. Use of HVDC interregional interfaces in the MT-HVDC scenario (Eastern
Interconnection) ..................................................................................................... 120
Figure C-1. Screenshot-generated custom tool developed for the NTP Study (Grid
Analysis and Visualization Interface) ..................................................................... 122
Figure D-1. WECC environmental data viewer ............................................................ 124
Figure D-2. BLM zone classes and transmission lines for Interregional AC Western
Interconnection case .............................................................................................. 124
Figure D-3. Costs associated with respective BLM zone ............................................ 125

List of Tables

Table I. Summary of Scenarios for Zonal-to-Nodal Translation .......................................ix
Table 1. Objectives of Nodal Modeling for the NTP Study ............................................... 2
Table 2. Different Use Cases for the Two Iterations of Scenarios Presented
in This Chapter........................................................................................................... 5
Table 3. Overview of Data Sources Used for Building CONUS Datasets ........................ 9
Table 4. Primary Models Used in the Transmission Expansion Phase of the Z2N
Translation ............................................................................................................... 10
Table 5. Summary of Scenarios for Zonal-to-Nodal Translation .................................... 21
Table 6. Summary of Common Themes Across Nodal Scenarios ................................. 29
Table 7. Summary of Differentiated Themes for Each Nodal Scenario .......................... 30
Table 8. Transmission Capital Cost for the Limited Scenario ........................................ 69
Table 9. Transmission Capital Cost for AC Scenario ..................................................... 69
Table 10. Transmission Capital Cost for the MT-HVDC Scenario .................................. 69
Table 11. Cost by Mileage and Voltage Class for the Limited Scenario ......................... 70
Table 12. Cost by Mileage and Voltage Class for the AC Scenario ............................... 70
Table 13. Cost by Mileage and Voltage Class for the MT-HVDC Scenario .................... 70
Table 14. Summary of Annual Savings Compared to the Limited Case ........................ 71
Table 15. Total Annualized Net Avoided Costs of the AC and MT-HVDC Scenarios
Compared to the Limited Scenario........................................................................... 72
Table 16. Detailed Generation and Revenue by Generator Type .................................. 73
Table 17. Disaggregation of Annual Benefits According to Stakeholders ...................... 75
Table A-1. Summary of Nodal Baseline Datasets for Contiguous United States
(CONUS) ................................................................................................................. 83
Table A-2. Overview of Data Sources Used for Building CONUS Datasets................... 83
Table A-3. Selection of Operating Conditions (“snapshots”) From Nodal Production
Cost Model for Use in DC Power Flow Transmission Expansion Planning Step ...... 89
Table B-1. Nodal Transmission Building Block Characteristics (overhead lines) ........... 94

National Transmission Planning Study

xvii

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Table B-2. Nodal Transmission Building Block Characteristics (transformation
capacity) .................................................................................................................. 94
Table B-3. Rationale for Nodal Transmission Portfolios (Limited scenario) ................. 104
Table B-4. Rationale for Nodal Transmission Portfolios (AC scenario) ........................ 107
Table B-5. Rationale for Nodal Transmission Portfolios (MT-HVDC scenario) ............ 110
Table D-1. BLM Cost per Acre by Zone Number ......................................................... 126
Table D-2. Required Width for Transmission Lines by Voltage Class .......................... 126
Table D-3. Land Cover and Terrain Classification Categories With Multiplier .............. 127
Table D-4. Cost per Mile by Voltage Class .................................................................. 127

National Transmission Planning Study

xviii

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

1 Introduction
The transmission system is a critical part of the electric power system that requires
detailed planning to ensure safe and reliable operations of the grid. A cornerstone of
power system planning for decades is nodal modeling and analysis. Nodal power
system models consist of the specific components of the system, such as generators,
loads, and transmission lines in mathematical representations to use for analysis and
simulation. To adequately evaluate the role and value of transmission for the contiguous
United States (CONUS), the National Transmission Planning Study (NTP Study) uses
nodal models for various analyses regarding power system planning and reliability. The
use of the nodal models in the NTP Study is part of a broader framework that applies
multiple models and integrates aspects of reliability and economics to evaluate how
transmission can help meet the needs of future systems. This chapter details the
methods for compiling detailed nodal models of the contiguous U.S. power system and
using this dataset to build transmission portfolios for the future as well as the
operational insights from production cost modeling of future U.S. grids.
Modeling the entire contiguous U.S. electricity system at a nodal level serves two
primary objectives: 1) it verifies the feasibility of future power systems that can meet the
physical constraints that dictate operations and 2) it provides insights on grid balancing
that match the temporal and spatial granularity with which grid operators view the
system. Table 1 further defines several subobjectives of nodal modeling. To achieve
these objectives, the study team combined industry planning models with several
scenarios adopted from zonal modeling of the CONUS using the Regional Energy
Deployment System (ReEDS) model, detailed in Chapter 2. 2 These future scenarios are
translated into nodal models by combining existing network details with plans for
generation, storage, and transmission. The result of this “zonal-to-nodal” (Z2N)
translation is a detailed nodal model of a future electric power system of the United
States. This resulting nodal model includes unit-level generation and node-to-node
transmission components, among many other details of the physical assets on the grid.
In the process of creating nodal models of the future, transmission portfolios are
developed that meet the needs of the power system and the intent of the scenario. The
development and verification of the transmission portfolios is a critical aspect of the
analysis that allows transformative scenarios to be tested against network and
operational constraints. The nodal models, inclusive of transmission portfolios,
produced with this exercise are used across multiple modeling efforts of the NTP Study,
some of which are detailed in Chapters 4 and 5 of this report.
Section 2 of this chapter presents the methods of the Z2N translation and the process to
plan the transmission portfolios. Section 3 presents the resulting transmission portfolios
of the contiguous United States for three scenarios for the model year 2035 and hourly
The ReEDS model contains 134 zones comprising the contiguous U.S. electricity system. ReEDS finds
the least-cost mix of generation, storage, and transmission to balance load and supply given various
technical, policy, and cost constraints for a set of years into the future. Chapter 2 details the long-term
planning scenarios developed with ReEDS, which is focused on the years out to 2050.
2

National Transmission Planning Study

1

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

operational insights. Section 4 presents analysis of the Western Interconnection of the
United States, including alternative transmission portfolios and operations, for a similar
set of scenarios in addition to an economic comparison of the investment and
operations of the scenarios from this section. Conclusions and potential future work are
discussed in Section 5.
Table 1. Objectives of Nodal Modeling for the NTP Study

Verify the feasibility of future scenarios by incorporating constraints that match
physical realities of operating power systems
-

The physical network model captures power flow distributions and loading patterns
across network elements (individual transmission lines and transformers).
Physical infrastructure limits are captured (individual equipment ratings).
Generators have distinct points of interconnection (POIs) and therefore drive related
transmission network upgrades.
The constraining of transmission flows aims to represent physical transmission
margins that emulate actual operations.
Enable more seamless data flow and obtain information to feed forward and feed
back to other modeling domains.

Gain grid-balancing insights based on detailed spatial and temporal modeling
-

-

Establish whether the system can balance load and generation at an hourly temporal
resolution considering both intraregional and interregional transmission network
constraints.
Identify which resources are serving load during stressful periods and the role played
by the transmission network to enable this.
Analyze the use of expanded interregional transmission and how this impacts system
operations.
Analyze potential intraregional network constraints and the resulting need for
upgrades and/or new investments.
Test the latent and potentially increased need for flexibility in parts of CONUS-wide
models (including the role of transmission in providing flexibility).
Assess the operation of energy storage to support load and generation balancing,
including the trade-offs with transmission network constraints.
Understand how much and where variable renewable energy (VRE) curtailment is
happening.

National Transmission Planning Study

2

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

2 Methodology
The following sections outline the methodology and workflows for the Z2N translation
and the additional analysis that is performed with the nodal models. This is illustrated as
a conceptual outline in Figure 1. The zonal input is a ReEDS scenario, which has 134
zones across the CONUS model. 3 The final nodal-level result is 1) a model year 2035
transmission portfolio that delivers broad-scale benefits and adheres to network
constraints and 2) a production cost model that can be used to understand the
operations at hourly resolution over a year (8,760 hours). Building the transmission
portfolios and the nodal production cost model involves two primary steps:
1. Disaggregation: The assets that ReEDS builds (generation and storage) are
disaggregated spatially across the United States within the zones and using
other underlying information on asset location, 4 and assets are assigned
points of interconnection (POIs) that already exist on the network.
2. Transmission expansion planning: New transmission is built within the
nodal network models to connect the generation and storage assets, and
additional transmission is built to connect regions and zones. The
transmission for connecting regions and zones uses ReEDS results as an
indication of need, but the exact transmission capacities from ReEDS are only
guides and not prescriptive.
This transmission planning phase uses a combination of automated methods and an
interactive transmission expansion planning approach to make decisions about discrete
transmission expansions. The resulting transmission portfolios and production cost
results are presented in this chapter and are used in Chapters 4 and 5 of this report for
subsequent analysis. The following section describes the Z2N translation process.

See Chapter 2 for more details on ReEDS and the zonal results.
Wind and solar are built within the ReEDS scenarios based on supply curves with discrete 11.5km x 11.5-km grid cells (for both wind and solar) across the United States. Therefore, these assets are at
a finer resolution than the 134 zones and are interconnected spatially in the network models according to
those grid points. See Chapter 2 for a more detailed explanation of ReEDS outputs.
3
4

National Transmission Planning Study

3

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 1. Conceptual outline of zonal-to-nodal translation
CEM = Capacity Expansion Model

National Transmission Planning Study

4

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

A note on using multiple software tools and workflows
Several of the results in this chapter originate from the earlier ReEDS scenarios,
which represent a version that was completed midway through the NTP Study.
These results are presented in Section 4. One of the primary purposes for using
these interim results is the downstream models, especially the alternating current
(AC) power flow models, can be very challenging and time-consuming to build and
using the earlier scenario results allowed for a full run-through of the multimodel
linkages—capacity expansion, production cost, and AC power flow—to be completed
within the timeline of the study. There are slight differences between earlier and final
scenarios (the scenarios analyzed in Chapter 2 are “final”), but both have the same
general trends with respect to generation and transmission capacity buildout to the
model year for this chapter: 2035. More details on the advancements made between
earlier scenarios and final ReEDS scenarios are summarized in Chapter 1, Appendix
A of this report.
The industry planning cases that are a starting point for the nodal analysis are the
same between earlier and final scenarios. In building the future year nodal database,
which requires the disaggregation and transmission planning steps outlined next, the
same general principles for the Z2N process are used. However, though
comparisons across modeling software of the same type (i.e., production cost) are
not done in this chapter or anywhere else in the study,1 there are still lessons from
analyzing each set of scenarios that can strengthen the lessons from the study,
which is the reason for including two sections of results. In addition, there is value to
industry in having multiple production cost databases and transmission portfolios
available as an outcome of the NTP Study, of which these will both be made
available to industry and others where possible. Table 2 gives a summary of how
earlier and final scenarios are used in this and other chapters.
Table 2. Different Use Cases for the Two Iterations of Scenarios Presented in This Chapter
Iteration of Zonal
ReEDS
Scenarios

Section for This
Chapter

Geographical Coverage
Represented in the
Models

Other Chapters That
Directly Use These
Capacity Expansion
Iterations

Earlier

Section 4

WI1

Chapter 4; Chapter 5

Final

Section 3

CONUS

Chapter 2

WI = Western Interconnection; EI = Eastern Interconnection; CONUS = Contiguous United States

National Transmission Planning Study

5

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

2.1 Zonal-to-Nodal Translation
The Z2N translation is focused on transmission expansions and production cost
modeling. The Z2N translation is a common process in industry, where resource plans
are often developed at a zonal scale, but those plans eventually must be mapped to a
nodal network model for a detailed transmission planning exercise. 5 There are many
challenges to this process because zonal models, where policies or other technoeconomic realities can be represented in detail, do not have the level of detailed
network and physics-based constraints that exist in nodal models.
For the NTP Study, the challenges in this process are magnified because 1) the nodal
datasets for the CONUS model(s) are very large, 2) the change with respect to the initial
grid configuration is substantial, and 3) there is significant diversity in resources and
network configurations across the United States, so strategies for the Z2N translation
must be flexible.
The compiling and building of the nodal models follow a few basic principles:
• Reproducibility between scenarios: There are many potential methods to
undertake the Z2N translation, as indicated by the multiple methods explained
later in this chapter. However, the methods are reproducible across scenarios.
Therefore, any comparisons of results in this chapter are across scenarios
developed using the same methods.
• Representing industry practice: Z2N should build from industry best data and
best practices. This starts with using industry planning models and learning from
industry practices in their latest planning studies and in consultation with
technical review committee (TRC) members. Section 2.1.1 details the starting
datasets.
• Using capacity expansion findings as a guideline: Although generation
capacity, storage, and demand are directly translated from the scenario-specific
ReEDS outcomes, the prescribed interregional transfer capacities are used as a
guideline for transmission expansion needs when building the nodal models.
An overview of the primary steps that form the entire Z2N translation is illustrated
in Figure 2, starting with the results from ReEDS (1); moving into the disaggregation of
zonally specified generation capacity, storage capacity, and demand into nodes (2);
followed by a set of transmission expansion steps (3). The underlying nodal datasets
that support this approach are further described in the next section.

5
For example, California Public Utilities Commission. 2022. “Methodology for Resource-to-Busbar
Mapping & Assumptions for The Annual TPP.” https://www.cpuc.ca.gov/-/media/cpucwebsite/divisions/energy-division/documents/integratedresource-plan-and-long-term-procurement-planirp-ltpp/2022-irp-cycle-events-and-materials/2023-2024- tpp-portfolios-and-modelingassumptions/mapping_methodology_v10_05_23_ruling.pdf

National Transmission Planning Study

6

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 2. Overview of zonal-to-nodal translation approach
Note: Figure incorporates iteration back to capacity expansion (ReEDS).
CEM = Capacity Expansion Model (ReEDS); PCM = Production Cost Model

2.1.1 Geographic extent and nodal datasets
The CONUS bulk power system model used for the NTP Study comprises three
regions—the Eastern Interconnection, Western Interconnection, and the Electric
Reliability Council of Texas (ERCOT), shown in Figure 3. These interconnections are
asynchronously interconnected through several relatively small back-to-back (B2B)
HVDC converters. 6 The geospatial scope of the NTP Study is the contiguous United
States. However, portions of Canada and Mexico are modeled but remain unchanged
with respect to resources and transmission infrastructure in the study. 7

There are seven B2Bs between the contiguous U.S. portions of the Eastern and Western
Interconnection (each ranging from 110 to 200 megawatts [MW]) and two between the Eastern
Interconnection and ERCOT (200 MW and 600 MW). Not shown in Figure 3 are two B2B converters
between Texas and Mexico (100 MW and 300 MW) as well as one between Alberta and Saskatchewan in
Canada (150 MW).
7
Industry planning models that are used for this study (see Table 3) include portions of Canada and
Mexico. The regions are still part of the production cost and power flow models used in this chapter, but
regions outside of the United States were not modified from the industry planning cases (i.e., load,
generation, or transmission was not added).
6

National Transmission Planning Study

7

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 3. Map of the three interconnections of the bulk U.S. power system
Note: Map highlights the interconnectivity between interconnections via existing B2B HVDC interties.

The nodal datasets used for the NTP Study are compiled from the industry planning
cases from the Western Electricity Reliability Council (WECC), 8 Eastern Interconnection
Reliability Assessment Group (ERAG), 9 and to some extent ERCOT 10—the details of
which are summarized in Table 3. These datasets typically do not have all relevant
details to inform a Z2N translation, i.e., geographical coordinates required to map zonal
capacity expansion results to individual nodes or run future scenarios beyond the
planning case timeline. Though WECC publishes a fully functional production cost
model as part of its Anchor Dataset (ADS), the ERAG and ERCOT available data lack
component information to build and run production cost models, such as generator unit
constraints (ramp rates, minimum stable levels, and minimum uptime and downtime,
among others), detailed hydro availability, or hourly wind and solar profiles. A task
within the NTP Study was designed to fortify the industry planning cases for the Eastern
Interconnection and ERCOT to be suitable for this study. The WECC ADS model details
were used directly for the CONUS scenarios. For the Western Interconnection analysis
Through the well-known WECC Anchor Dataset (ADS) (Western Electricity Coordinating Council 2022).
The Eastern Interconnection Reliability Assessment Group (ERAG) oversees the Multiregional Modeling
Working Group (MMWG), which is responsible for assembling nodal network models for the Eastern
Interconnection (among other responsibilities).
10
WECC is the regional entity responsible for reliability planning and assessments for the Western
Interconnection and has nodal models of the Western Interconnection available for members. EIPC is a
coalition of regional planning authorities in the Eastern Interconnection that compiles a power flow model
of the Eastern Interconnection based on regional plans. ERCOT did not provide an updated power flow
model from this study; therefore, data were compiled from publicly available sources and purchased data.
8
9

National Transmission Planning Study

8

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

of the earlier scenarios, an augmented model was developed to include five future
transmission projects. For details on the development of the Western Interconnection
database, see Konstantinos Oikonomou et al. (2024).
The nodal datasets of the Eastern Interconnection, Western Interconnection, and
ERCOT are each developed independently and then combined to form the CONUS
nodal dataset. A summary of sources for the data in the nodal models is presented
in Table 3, with a more comprehensive table in Appendix A.1. Some of the benefits
realized and challenges faced in compiling a CONUS-scale dataset are summarized in
Appendix A.2.
Table 3. Overview of Data Sources Used for Building CONUS Datasets
Eastern
Interconnection

Western
Interconnection

ERCOT

ERAG MMWG 20312

WECC ADS 2030 v1.5

EnergyVisuals5

NARIS, MapSearch,
EnergyVisuals,
EIA 860

NARIS MapSearch,
EnergyVisuals, EIA 860

NARIS,
MapSearch,
EnergyVisuals

NARIS, EIA 860,
EIPC

WECC ADS 2030 v1.5,
EIA 860

NARIS

NARIS, EIA CEMS

WECC ADS 2030

NARIS

Description
Network topology (node/branch
connectivity)1
Node mapping (spatial)
Generation capacity
(technology)
Generation techno-economic
characteristics3
1

Augmented through stakeholder feedback to include the most recent available data on network updates/additions.

2

Eastern Interconnection Reliability Assessment Group (ERAG) Multiregional Modeling Working Group (MMWG) 2021
series (2031 summer case).
3

Includes heat rates, minimum up-/downtimes, ramp rates, and minimum stable operating levels.

4

Hourly/daily/monthly energy budgets (as appropriate).

5

Power flow case files (2021 planning cases).

ADS = Anchor dataset (Western Electricity Coordinating Council 2022); CEMS = continuous emission monitoring
system; EER = Evolved Energy Research, MMWG = Multiregional Modeling Working Group; NARIS = North American
Renewable Integration Study (National Renewable Energy Laboratory [NREL] 2021a); WECC = Western Electricity
Coordinating Council

2.1.2 Modeling domains and tools
Several modeling domains are applied as part of the iterative Z2N translation process.
For disaggregation, many custom tools were built to map zonal resources and network
assets. A summary of the primary modeling tools for the transmission expansion phase
is shown in Table 4.

National Transmission Planning Study

9

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Table 4. Primary Models Used in the Transmission Expansion Phase of the Z2N Translation
Step in Transmission Planning Phase of
Translation

Primary Modeling Tools

1. Production cost

Sienna, 11 GridView 12

2. Power flow and contingency

PSS/E, 13 PowerWorld Simulator 14

3. Visualization

QGIS, 15 Grid Analysis and Visualization
Interface 16

The outputs of the ReEDS scenarios are used as inputs into the disaggregation step.
The objective of disaggregation is to translate new electricity demand, generation, and
storage that is built in the scenario from ReEDS to augment an established industry
planning case to build scenario-specific nodal production cost models. Section 2.2
details this process.
However, not all resources are easily placed on the network with automated processes,
and additional expertise is needed to make decisions about resource allocation to
nodes. Interventions are needed for some specific instances, where the resource—for
example, wind—might be in such high quantities that a substation or entire highvoltage/extra high-voltage (HV/EHV) collector network 17 might need to be considered.
These interventions are where the transmission expansion planning phase begins. The
software and tools in Table 4 are used to both ensure generation and storage have
sufficient transmission to interconnect to the network and transmission portfolios are
designed to meet the needs of the system and reflect the intention of the ReEDS
scenario. Several times throughout the design of the transmission portfolios, production
cost models and DC power flow models are run and analyzed to assess how the system
operates—that is, energy adequacy, hourly balancing of supply and demand, variable
renewable energy (VRE) curtailment, individual branch loading, and interface flows.
Section 2.3 details this phase of the Z2N process.
The nodal production cost models are the core analytical tools used in the Z2N
process—employed extensively in the transmission expansion planning phase and
producing the operational results analyzed for Sections 3 and 4. They also seed the
operating points for the more detailed reliability analysis described in Chapter 4 and
Chapter 5. The open-source modeling framework, Sienna/Ops, is used as the
Sienna: https://www.nrel.gov/analysis/sienna.html, last accessed: February 2024.
GridView: https://www.hitachienergy.com/products-and-solutions/energy-portfoliomanagement/enterprise/gridview, last accessed: May 2024.
13
Siemens PSS/E, https://www.siemens.com/global/en/products/energy/grid-software/planning/psssoftware/pss-e.html, last accessed: February 2024.
14
PowerWorld Simulator, https://www.powerworld.com/, last accessed: March 2024.
15
QGIS, https://www.qgis.org/en/site, last accessed: May 2024.
16
For the final scenarios, the study authors developed the Grid Analysis and Visualization Interface to
help visualize the system operations and support transmission planning. Details are presented in
Appendix C.2.
17
As defined in ANSI C84.1-2020 where HV = 115–230 kilovolts (kV) and EHV = 345–765 kV.
11
12

National Transmission Planning Study

10

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

production cost model for transmission planning and operational analysis for final
scenarios (Section 3 of this chapter) whereas GridView 18 is used for earlier scenarios
(Section 4 of this chapter).

2.2 Disaggregation
The Z2N disaggregation relies on two principles about how to convert ReEDS results to
nodal models: 1) generation and storage planned in ReEDS must align with the nodal
model very closely (i.e., prescriptive) and 2) transmission results from ReEDS are
indicative of the needed transmission capacity to move energy between regions.
Demand is also prescriptive because it is an input to the zonal model. The distinction
between the prescriptive and indicative use of zonal results is summarized in Figure 4.

Figure 4. Overview of the zonal results flowing into the nodal model
LPF = load participation factor

The prescriptive component of the Z2N process comprises four primary steps:
1. Demand disaggregation: Demand disaggregation is needed to represent load
profile time series in nodal production cost models because only zonal demand is
provided by ReEDS outcomes. ReEDS zonal demand is disaggregated into
nodes for each ReEDS region based on load participation factors (LPFs) 19
derived from the industry planning power flow models. This process is depicted
geospatially in in Figure A-38 in Appendix A.3, demonstrating how a single zonal

GridView: https://www.hitachienergy.com/products-and-solutions/energy-portfoliomanagement/enterprise/gridview, last accessed: May 2024.
19
Load participation factors (LPFs) are a measure to assess the contribution of an individual load to
overall load in a region. In the context of the NTP Study, LPFs are derived from absolute nodal loads in
the nodal datasets to generate load profiles from ReEDS zonal loads for nodal production cost model
analysis.
18

National Transmission Planning Study

11

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

demand profile from a ReEDS zone is disaggregated into the many nodes based
on the relative size of each load (their LPFs).
2. Establish the generation and storage capacity needs for each zone in the
nodal model year: To establish the amount of generation and storage to add to
the system, the study team compared industry planning cases with the ReEDS
scenario. Generation and storage capacity is added or subtracted (i.e.,
deactivated) to the nodal models to reach prescribed capacities defined by
ReEDS. 20
3. Ranking of nodes—POIs: To assign zonal capacities to nodal models, the study
team ranked nodes by generation and storage capacity POI favorability. Viable
nodes 21 are ranked by build favorability. Higher build favorability considers the
following (in descending order): 1) nodes with more deactivated/retired capacity,
2) nodes at higher voltage levels, 3) nodes or surrounding nodes with large loads
attached. 22
4. Adding generation and storage capacity: The final step in the disaggregation
is to link new generation and storage capacity defined in ReEDS to actual nodes,
or POIs. Figure 5 illustrates the spatial clustering of solar photovoltaics (PV) and
land-based wind to POIs. After initial clustering to POIs, further refinement may
be required to ensure node capacities are reasonable given network constraints.
Appendix A.3 contains further details on this step.

ReEDS compiles any known generator retirements from EIA or other sources and exogenously
enforces them. In addition, ReEDS can retire generation economically. See Chapter 2 for more details.
21
Nonviable nodes may be terminals of series capacitors, tap-points, fictitious nodes to indicate
conductor changes (or three-winding transformer equivalents) and disconnected/isolated busses.
22
For allocating distributed energy resources (distributed solar PV), which is consistent across all NTP
scenarios (sensitivities on types of distributed energy resources, quantity or distribution of distributed
energy resources is out scope for the NTP Study).
20

National Transmission Planning Study

12

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 5. Illustration of VRE disaggregation process as part of Z2N translation
Note: The VRE disaggregation is demonstrated for solar PV (left) and land-based wind (right) where individual solar PV and wind
sites are assigned to POIs (nodes).

2.3 Transmission Expansion
Nodal transmission expansion comprises a key part of the overall Z2N translation
workflow. The scenarios for the NTP Study are transformational, in part because of the
decarbonization targets enforced in several of them and the scale of transmission
allowed. Therefore, the study team developed a transmission expansion process that
could meet the demands of potentially large buildouts in interregional and regional
networks. In addition, this process needed to be manageable at an entire
interconnection and contiguous U.S. scale. This manageability is achieved through
stages of increasing complexity to maintain tractability as the transmission portfolios are
built out in the nodal production cost model. This staged process is summarized in
Figure 6. Each stage becomes progressively more constrained and can have several
internal iterations per stage.

National Transmission Planning Study

13

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 6. Stages of transmission expansion planning
Orange boxes indicate a nodal production cost model, teal boxes indicate DC power flow, and blue boxes indicate transmission
expansion decisions by a transmission planner.

The transmission expansion planning process applies the staged principles to remain
consistent across CONUS and is not intended to exactly replicate transmission planning
processes or adopt regionally specific technical transmission guidelines/standards as
applied by regional transmission planners, owners/developers, or utilities. In addition, to
help visualize the large changes and impacts to the Z2N scenarios, the Grid Analysis
and Visualization Interface tool was created to visualize hourly production cost model
results (see Appendix C.1 for an overview of this tool).
The sections that follow provide further details of the staged transmission expansion
process depicted in Figure 6.
Stage 1: Initial network performance assessment
Stage 1 is a performance assessment intended to provide an initial analysis of system
operations and transmission network use before transmission expansion is undertaken.

National Transmission Planning Study

14

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

The initial disaggregation of resources has already been implemented when this first
stage of transmission expansion planning is reached (see Section 2.2). Therefore, the
Stage 1 nodal production cost model runs contain all the generation and storage
resources built in the ReEDS scenario.
Stage 1 is implemented by a nodal PCM with two core transmission network
formulations: 23
• Transmission unconstrained: No transmission interface or branch bounds are
applied (analogous to a “copper sheet” formulation but with network impedances
represented to establish appropriate flow patterns)
• Transmission interface constrained: Only transmission interface limits are
enforced (individual tie-line branch limits are unconstrained). 24
The unconstrained nodal production cost model establishes where power would want to
flow without any network constraints and hence establishes an idealized nodal
realization without network bounds. The semibounded interface-constrained nodal
production cost model establishes an indicator for interregional and enabling
intraregional transmission network needs by allowing individual branch overloading but
ensuring interregional tie-line flows remain within the limits established from the ReEDS
transfer capacities (aggregated to transmission planning regions). 25
At this stage, the following system performance indicators are assessed (for both
transmission-unconstrained and transmission interface-constrained formulations):
• Interregional power flow patterns across sets of transmission interfaces (tie-lines)
• Individual tie-line power flow patterns and loading profiles
• Individual HV and EHV transmission line loading profiles
• Regional wind and solar PV curtailment levels
• Operations of dispatchable resources.
After establishing general flow patterns within and between regions in Stage 1, the
study team proceeded with Stage 2.

Further details on transmission formulations are provided in Appendix A.4.
Transmission interfaces are defined as the sum power flows across each individual tie-line part of the
given interface.
25
Using constrained and unconstrained production cost models as a way to gather information about the
system potential is typical in industry planning, for example, in Midcontinent Independent System
Operator (MISO) (Dale Osborn 2016).
23
24

National Transmission Planning Study

15

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Stage 2: Iterative transmission expansion planning for normal operations (system
intact)
Stage 2 expands the HV and EHV transmission networks for normal operating
conditions (system intact). To plan the transmission expansion, an iterative approach is
deployed, which includes the following steps:
1. Performance assessment: From a set of nodal production cost model results
(Stage 1 transmission interface-constrained or a Stage 2 iteration), assess network
use metrics such as interface flow duration curves, loading duration curves of
individual tie-lines, individual HV and EHV transmission line loading profiles, VRE
curtailment, dispatch of resources (generation, storage), and unserved energy.
2. Transmission expansion planning: Based on the interface flow patterns,
iteratively propose and test nodal transmission expansion options to increase
transmission capacities while managing HV and EHV network overloads. The quick
assessment of the feasibility of the proposed transmission expansion options can
be obtained by employing DC power flow 26 simulations as an intermediate step
prior to running a new nodal production cost model simulation including the
additional transmission network elements.
3. Nodal production cost model: Once a substantial set of new transmission
expansions meets performance metrics in the DC power flow simulations, the
nodal production cost model is adapted with the new transmission and run with an
increasing set of network constraints.
In each iteration of the Stage 2 transmission expansion process described previously,
different transmission expansion priorities are addressed. Figure 7 shows the priority
order followed for the definition of the network expansion options defined in Stage 2.
Once the Stage 2 transmission expansion planning process is complete, the system
operation performance is assessed using a constrained nodal production cost model
formulation incorporating all the proposed transmission expansion portfolios. Given
good performance in the nodal production cost model, this establishes the starting point
for Stage 3. Criteria considered in evaluating production cost model performance are
unserved energy, new transmission utilization, and VRE curtailment.

26
DC power flow simulations are performed over a subset of the 8,760 operating conditions from the
nodal production cost model results. These representative operating conditions are called “snapshots.”
The selection of these snapshots is made to capture periods of high network use from different
perspectives, such as peak load conditions, high power flows across interfaces, and between
nonadjacent regions. See Appendix A.6 for an outline of periods selected as snapshots.

National Transmission Planning Study

16

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 7. Prioritization of transmission expansion planning approach

Stage 3: Transmission expansion planning with selected single transmission
network contingencies
Stage 3 of the Z2N process applies similar logic to that of Stage 2 but with the system
intact linear power flow analysis being replaced by selected single contingency linearpower-flow-based contingency analysis. Contingencies are defined as individual branch
element outages, 27 which include the subset of branches that form tie-lines between
transmission planning regions 28 plus all lines within a transmission planning region that
directly connect to a tie-line (“first-degree neighbors”) inclusive of any newly added tielines between regions (as shown in Figure 8).
The planning criteria applied in the final scenarios’ CONUS expansion results are the
monitoring of all branches greater than or equal to 220 kV, postcontingency overload
threshold of 100% of emergency ratings (“Rate B”) for lines not overloaded in the
precontingency state, and postcontingency loading change greater than 30% (relative to
the precontingency state) for lines already overloaded in the precontingency state. 29 In
the earlier scenario results for the Western Interconnection (results shown in Section 4),
the emergency rating is used for postcontingency flows. When this is not available,
Analogous to P1 and P2 type contingencies defined in NERC TPL-001-5 (North American Electric
Reliability Corporation [NERC] 2020).
28
The regional boundaries between transmission planning regions and the underlying balancing
authorities (BAs) in industry are slightly different from those used in the NTP Study. Chapter 2 maps the
assumptions for the geographic bounds of the regions used throughout the NTP Study.
29
In cases where a transmission line under the contingency list is modeled as a multisegment line or
includes a tap point without any load or generator connected to it, the selected single contingency is
modeled to include the tripping of all segments composing the model of the given single contingency. In
other words, a single contingency is modeled by tripping multiple elements simultaneously. In cases
where Rate B = Rate A or Rate B = 0 in datasets, the postcontingency threshold is set to a default of
120% of Rate A (135% for earlier scenario analyses).
27

National Transmission Planning Study

17

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

135% of normal rating is used. Because all branch constraints for branches 230 kV and
above are enforced, no additional threshold is provided for precontingency overloaded
branches. Decisions on expanding the network following the linear (DC) contingency
analysis are based on consistent and/or widespread network overloads.

Figure 8. Illustration of single contingency definition for Stage 3 of zonal-to-nodal process
The selected single contingencies that comprise the contingency analysis are composed of tie-lines (red) and first-degree
neighbors (blue).

Additional network expansions following the DC contingency analysis results are based
on the principles depicted in Figure 9. In situations where a given contingency causes
consistent and widespread network overloads, transmission reinforcements around the
given contingency are proposed (mitigation of the cause). In situations where a given
network element(s) is consistently overloaded across multiple contingencies and/or
multiple operating conditions, network reinforcement solutions around the impacted
element(s) are proposed (mitigation of the consequences).

National Transmission Planning Study

18

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 9. Illustration of mitigation of causes and consequences during contingency analysis

The study team performed contingency analysis on all existing and new interregional
tie-lines as illustrated in Figure 8 and Figure 9. The contingency analysis is intended as
a screen to address large interregional contingency risk considering some ties (500 kV
and 765 kV in particular) have the potential to carry large amounts of power—on the
order of 3000–5000 MW. However, the contingency analysis is not a comprehensive
intraregional contingency assessment, so further analysis of new transmission
expansions would be necessary when moving toward detailed network design stages,
which is beyond the scope of the NTP Study. 30

2.4 Transmission Planning Feedback to Capacity Expansion
The NTP Study approach included a feedback loop between zonal ReEDS capacity
expansion findings and downstream nodal models to capture key learnings from initial
capacity expansion findings and resulting downstream nodal model findings, illustrated
in Figure 10. During the initial modeling for the NTP Study, the team had several rounds
The results for Section 4 use a variation of the contingency method illustrated here. See Appendix A.7
for an explanation of this method, which was applied to the Western Interconnection earlier scenarios.
30

National Transmission Planning Study

19

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

of scenarios that were used to refine these multimodel linkages and feedback
constraints to ReEDS based on the buildout of nodal models. This feedback mainly
focused on the following:
1. Improving the spatial distribution of wind and solar resources
2. Constraining maximum interzonal transmission buildout capacities.
For Item 1, the study team recognized that regions with large interregional transmission
would in some cases require substantial intraregional network strengthening to
accommodate large transfers to other regions. The result of the feedback is an
improved representation of spur and reinforcement costs in the ReEDS final NTP Study
scenarios, which is described further in Chapter 2. As an indirect effect of this
improvement, concentrating large amounts of generation and transmission
infrastructure in small geographical footprints is lessened.
For Item 2, the feasibility of certain high-capacity corridors in the earlier rounds of zonal
capacity expansion modeling results was flagged by TRC members and the study team
as being difficult to implement. In developing transmission portfolios of these corridors in
the Z2N translations, several technical challenges were revealed with respect to the
amount of transmission capacity that could be practically realized between ReEDS
regions (both spatially and electrically). As a result, an upper bound of 30 gigawatts
(GW) of maximum transmission buildout across a given corridor was set for the final
round of ReEDS scenarios.

Figure 10. Overview of transmission expansion feedback between zonal and nodal modeling domains

National Transmission Planning Study

20

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

2.5 Scenarios for Nodal Transmission Plans and Production Cost
Modeling
A subset of the 96 scenarios from ReEDS was chosen to be translated into nodal
scenarios because of the complexity of working with large nodal power system models.
Therefore, the study team narrowed down the scenarios with the intent to understand
distinct transmission expansion learnings between each scenario, interest from the
TRC, and feasibility of implementation. The study team opted for scenarios situated
between the extremes of cost and demand projections as well as technology
advancements. In addition, though the scenarios from ReEDS extend to 2050, the nodal
models for this chapter are focused on 2035 to understand the more immediate
implementation challenges. Table 5 summarizes the three scenarios for the
development of nodal interregional transmission expansion.
Table 5. Summary of Scenarios for Zonal-to-Nodal Translation

Dimension

Limited

AC

MT-HVDC

Transmission framework1

AC expansion within
transmission planning
regions

AC expansion within
interconnects

HVDC expansion across
interconnects
(+AC within transmission
planning regions)

Model year

Annual electricity demand

CO2 emissions target
1

2035
Mid Demand1
CONUS: 5620 TWh (916 GW)
Western Interconnection: 1097 TWh (186 GW)
ERCOT: 509 TWh (93 GW)
Eastern Interconnection: 4014 TWh (665 GW]
CONUS: 90% reduction by 2035
(relative to 2005)

See Chapter 2 for further details.

CO2 = carbon dioxide; AC = alternating current; TWh = terawatt-hour; GW = gigawatt; HVDC = high-voltage direct current

A summary of the interregional transmission expansion from the ReEDS zonal
scenarios is shown in Figure 11 (aggregated to the subtransmission planning region
level). 31 Maps of the results of zonal transmission expansion from ReEDS for the three
scenarios are shown in Appendix B.4. Large transmission planning regions are split into
meaningful subregions for transmission expansion, analysis, and insights from zonal
ReEDS findings.

All results in this chapter from the zonal ReEDS capacity expansion are analyzed in greater detail in
Chapter 2. Chapter 2 highlights more 2050 results, although Appendix B of Chapter 2 provides an
overview of the 2035 results.
31

National Transmission Planning Study

21

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 11. Interregional transfer capacity from ReEDS zonal scenarios used for nodal Z2N scenarios

National Transmission Planning Study

22

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

2.6 Economic Analysis of the Western Interconnection (earlier
scenario results)
This economic analysis evaluates the avoided costs and disaggregates economic
benefits among different network stakeholders of the nodal scenarios from the earlier
ReEDS expansions. The Limited scenario is used as a reference to estimate the
avoided cost of the AC and MT-HVDC cases. This study uses avoided costs as the
metric to estimate the economic benefits of transmission capital, generation capital, and
operation. Avoided cost is an accurate measure of the economic benefits when load
demand is constant across the scenarios analyzed (Hogan 2018; Mezősi and Szabó
2016). See Appendix D.4 for a brief discussion of prior studies that have estimated the
economic benefits of transmission. The production cost modeling uses the same
demand as an input to the scenarios modeled, so this condition is met for the scenarios
in this study. The overall avoided costs are estimated as the sum of the capital and
operational avoided costs after they have been annualized to enable comparison of
avoided costs that occur over different time horizons. This section discusses the
methodology of how the capital costs of generation and transmission were developed.
The section also describes the methodology for estimating the total avoided costs using
annualized values as well as the methodology used to disaggregate the avoided costs
as benefits among different network users. Disaggregating the benefits can help
understand how the avoided costs enabled by interregional transmission could
economically benefit different stakeholders, providing information to the multiple groups
that must collaborate to build transmission infrastructure.
2.6.1 Transmission capital cost methodology
Transmission costs are based on the transmission lines added to the GridView
Production Cost Model by voltage class (230 kV, 345 kV, and 500 kV) according to the
methodology introduced in the previous sections. The study team used the WECC
Transmission Calculator (Pletka, Ryan et al. 2014) that was updated by E3 in 2019 (E3
2019) to calculate the capital costs for transmission. New generation is added to viable
POIs at 230 kV and above. If no POI is sufficiently close, 32 a new POI is created. As
such, a significant portion of what might be called “spur lines” in the capacity expansion
model is explicitly modeled in the nodal builds. The WECC Calculator multipliers for
land ownership and terrain were used to estimate the cost of added transmission along
with the costs associated with the allowance for funds used during construction
(AFUDC). 33 Appendix D.3 provides more detail on the land ownership and terrain
multipliers.
2.6.2 Generation capital cost methodology
Generation capital costs are taken directly from ReEDS outputs from earlier scenarios
for 2022–2034 for Limited, AC, and MT-HVDC. The capital cost inputs to ReEDS for
each generation type are taken from the Annual Technology Baseline (ATB)
For all wind and solar locations in the AC and Limited (Lim) scenarios, the average distance to their
POIs is 17 miles.
33
AFUDC is the cost of money invested or borrowed during construction that must be accounted for in
the total costs of construction.
32

National Transmission Planning Study

23

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

database (National Renewable Energy Laboratory 2023). The values are in 2004$ and
were escalated to 2018$ using the Consumer Price Index to match year dollars in
GridView. Costs are not split between the resource types, meaning all generation capital
costs are aggregated to a single dollar number.
2.6.3 Operating cost economic metrics methodology
The total economic benefits are estimated as the avoided cost for grid operation, across
the different transmission topologies. The costs modeled in GridView include fuel,
startup, shutdown, and other variable operating costs. The relative avoided costs of the
AC and MT-HVDC transmission topologies can be calculated by subtracting the total
cost of the AC and MT-HVDC scenarios from the Limited scenario:
Avoided Cost 𝑖𝑖 = Cost 𝐿𝐿𝐿𝐿𝐿𝐿 − Costsi 𝑖𝑖 ∈ {AC, MT-HVDC}.

The study team estimated the subsidy payments for the investment tax credit (ITC) and
production tax credit (PTC). The ITC applies a 30% credit to generation capital costs for
solar plants 34 (U.S. Department of Energy [DOE] 2022a) and the PTC applies a
$26/MWh credit to wind generation (DOE 2022b).
2.6.4 Net annualized avoided cost methodology
To account for the difference in timing and useful lifetime of the different capital
investments, the study team annualized the avoided transmission capital costs and the
avoided generation capital costs. The net annual avoided costs are equal to the sum of
the annual avoided operation costs and the annualized avoided capital costs. When net
annualized avoided costs are greater than zero, the scenario should be considered to
have positive overall avoided costs. The following formula is used to compute the
annualized cost:
𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴 𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶 = 𝐶𝐶 ×

𝑟𝑟(1 + 𝑟𝑟)𝑛𝑛
(1 + 𝑟𝑟)𝑛𝑛 − 1

Where r is the discount rate and n is the assumed lifetime of the infrastructure. The
study team used discount rates of 3% and 5% to show the net annualized value across
a range of discount rates. The assumed life of transmission capital is 40 years, and the
assumed life of generation capital is 20 years. Annualized value is also sometimes
called the annuity value and can be interpreted as the annual payment required over the
lifetime of an investment to be equivalent to the net present value of an investment paid
today as a lump sum (Becker 2022).
2.6.5 Benefit disaggregation and calculating the annualized net present value
In this chapter, the study team disaggregated system benefits according to different
network users, such as the consumers, the producers, and transporters. For the grid,
the generators are the producers, transmission owners are the transportation entity, and
power purchasers are the consumers. The NTP Study does not model the final stage of
These use round 1 ReEDS results where solar is assumed to take the ITC. This does differ from round
2 results where solar takes the PTC.
34

National Transmission Planning Study

24

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

the grid where the power purchasers distribute load to the end consumers through the
ratemaking process. This ratemaking process varies across states and utilities, but in
general costs (and avoided costs) are passed on to consumers through this process.
Disaggregating benefits according to network users provides additional information that
could enable more efficient planning to utilities, independent system operators (ISOs),
and system operators. For a single decision maker, such as a vertically integrated utility,
the optimal transmission topology would minimize system cost required to serve load
demand subject to regulatory requirements. In contrast, in a pure market setting, the
optimal transmission topology would be consistent with the market equilibrium where
market participants (network users) take prices as given (Hogan 2018). The electric grid
is a hybrid of these two structures. Many regions are served by a single utility; however,
interregional transmission typically requires collaboration across multiple utilities, ISOs,
and local stakeholder groups. In addition, wholesale electricity markets are becoming
more prevalent, and the frequency of power purchase agreements (PPAs) between
generators and utilities or consumers is increasing. Understanding the relative benefits
to different network users can help achieve a broader consensus among network users
that must cooperate to build interregional transmission (Kristiansen et al. 2018).
A critical component required to disaggregate economic benefits is an estimate of the
market price. Electricity price forecasts typically use historical data on electricity prices
and other variables. This can result in high forecast error as the forecast progresses
farther into the future and as other conditions change (Nowotarski and Weron 2018).
The most common approach for estimating prices from PCM simulations is to use the
locational marginal prices estimated by the model. However, the simulated LMPs
estimate revenues that are insufficient for cost recovery for most generators in the
model. The scenarios simulated the 2035 grid with significantly higher renewable
penetration than at present, making a forecast using the simulated locational marginal
prices, or present and historical prices unreliable. To alleviate this issue, the study team
assumed each generator receives its levelized cost of energy (LCOE) as its price,
approximating the lower bound that the generators would need, on average, to attract
investment financing. LCOE values were obtained from the ATB dataset for land-based
wind, offshore wind, solar PV, geothermal, battery storage, and hydropower using
moderate estimates, research and development (R&D) financials, and the year 2035.
LCOE values were obtained from Lazard (2023) for nuclear, coal, and natural
gas (Lazard 2023). This assumption models a generator operating under a PPA at a set
price or a price that is steadily escalating with inflation over time. LCOE represents the
minimum average price that a generator project requires to attract capital
investment (Lai and McCulloch 2017). This assumption ensures the modeled price is
sufficient for investment in the modeled generation capacity to occur. Of note, the
disaggregation of benefits and the modeling assumptions for prices do not affect the
total benefit estimates. Changes in prices will cause a transfer of benefits between
generators and power purchasers but does not affect the total benefits.
Generator benefits are estimated using the profits (revenue minus cost) to the
generators. The generator costs were obtained from the GridView simulations.
Generator revenue is estimated by multiplying the price (LCOE) by the dispatched
National Transmission Planning Study

25

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

generation—that is, generators are not compensated for generation curtailed in the
model. Because LCOE is the minimum average price required, this estimate of
generator benefits represents a lower bound on the annual generator benefits that
would be feasible to serve the required load. The total benefits to generators are
computed by summing over all generators using the following equation:
𝐽𝐽

𝐽𝐽

𝑗𝑗=1

𝑗𝑗=1

𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝑡𝑡𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔𝑔 =   𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝑡𝑡𝑗𝑗 =   𝐿𝐿𝐿𝐿𝐿𝐿𝐸𝐸𝑗𝑗 × 𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝑛𝑛𝑗𝑗 − 𝐶𝐶𝐶𝐶𝐶𝐶𝑡𝑡𝑗𝑗

Transmission owner benefits are estimated as the annual revenue given to the
transmission owners. In practice, transmission owners are remunerated through a
variety of payment mechanisms that vary across jurisdictions. The study team did not
model a formal transmission sale market, so the team used the annual revenue
requirement for the transmission capital. The annual revenue requirement is the amount
of revenue that a capital investment requires to attract the initial financial capital needed
to build the infrastructure (Becker 2022). This ensures the transmission owners receive
sufficient revenue to build the transmission capital in the modeled scenarios. Similar to
using LCOE to model generator revenue, this assumption represents a lower bound on
the annual benefit for transmission owners. Annual revenue requirements are computed
using the following equation:
𝑟𝑟(1 + 𝑟𝑟)𝑛𝑛
𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝐵𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡 𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑜 = 𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇 𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶 𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶 ×
(1 + 𝑟𝑟)𝑛𝑛 − 1

Where 𝑟𝑟 is the discount rate and 𝑛𝑛 is the expected lifetime of the infrastructure. The
study team used 𝑟𝑟 = 5% and 𝑛𝑛 = 40 years for transmission capital.

The study team assumed the power purchaser benefits are equal to the total load
benefits minus total load payments. The total load benefits represent the overall
economic benefit to end consumers from all electricity uses. The total load benefit is a
large quantity that is outside the scope of this study to estimate. When the total load
benefit is equal across the scenarios being compared, taking the difference to estimate
the relative benefits of the scenarios results in the total benefit term exactly to zero.
When this holds, the relative power purchaser benefit simplifies to the avoided cost in
load payments. The critical assumption required for the total load benefit to be equal
across scenarios is that load demand is the same across the cases being compared,
and this assumption is held true within the production cost modeling input assumptions.
The power purchaser payment is the sum of payments to generators and payments to
transmission owners.
The final stakeholder considered are the taxpayers. The PTC subsidy is paid for broadly
by the taxpayer base. This study does not model any formal tax policy regarding tax
brackets and assumes taxes are paid when the subsidies are paid. This avoids
assumptions about whether the subsidies are paid through government debt or some
other mechanism. This simplifies the analysis so the taxpayer benefit is the avoided

National Transmission Planning Study

26

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

cost of subsidy payments. If tax payments increase between scenarios that are
compared, this manifests as a negative benefit to the taxpayers.
The annual benefit for each network user for each scenario is the difference between
the AC and MT-HVDC transmission scenarios compared to the Limited scenarios; i.e.,
for each network user j, their benefit is calculating using:
Benefit 𝑖𝑖,𝑗𝑗 = Cost 𝐿𝐿𝐿𝐿𝐿𝐿,𝑗𝑗 − Costsi,j 𝑖𝑖 ∈ {AC, MT-HVDC}.

National Transmission Planning Study

27

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

3 Contiguous U.S. Results for Nodal Scenarios
This section presents the study team’s results of the nodal transmission portfolios for
the contiguous United States for the model year 2035, with emphasis on the resulting
transmission plans and the operations of these systems. 35 The study team finds there
are several transmission portfolios that can meet the hourly balancing needs of a high
renewable energy scenario and, with increased interregional transmission, imports and
exports can have large impacts on some regions—resulting in exchanges that exceed
the total demand of these regions.

3.1 A Range of Transmission Topologies Enables High Levels of
Decarbonization
The Z2N translation of ReEDS scenarios demonstrates how a range of transmission
topologies and technologies can be used to meet long-term resource changes. This is
demonstrated in three scenarios modeled at a nodal level—Limited, AC, and MT-HVDC
(and summarized in Table 6 and Table 7). 36 Transmission provides benefits for reaching
high levels of decarbonization in a country with substantial amounts of regional diversity
in generation resources and across scenarios that analyzed a range of strategies for
system growth, transmission topologies, and technology choices. The three nodal
scenarios demonstrate interregional or intraregional work in tandem to enable power to
move around the country and substantial amounts of both types of transmission are
seen in the transmission expansions.
In the Limited scenario, transmission is maximized at the regional level, building on
existing high-voltage alternating current (HVAC) networks in the regions. The AC
scenario portfolios find local transmission is still important and in need of expansion, but
interregional transmission plays a much bigger role—especially in areas of the country
where large amounts of new resources are added or where existing HV transmission
networks are less developed and interconnected. The need for large expansions of
interregional transmission in some regions necessitates voltage overlays (Southwest
Power Pool [SPP], MISO, WestConnect) but, in many instances, expansions using
existing voltage levels is sufficient.
The MT-HVDC scenario is a very different paradigm compared to systems dominated
by HVAC transmission expansion. The HVDC expansions are all long-distance and
connecting between neighboring regions, or farther. In addition, the HVDC is largely
embedded within AC networks that do not have any experience with HVDC, signifying
new operational frameworks would need to be created to handle a very different
operating regime. Some regions—such as SPP, MISO, WestConnect, and ERCOT—
have substantial enough buildouts of HVDC that it could play a large role in balancing
supply and demand. Considering the scale of HVDC expansion envisioned in this study,
35
All results in Section 3 are from the final ReEDS scenarios, which are inclusive of Inflation Reduction
Act impacts and other advances in the ReEDS model. See Chapter 2 for more details on the final ReEDS
scenarios.
36
Further detailed findings from each scenario are provided in Figure B-45, Figure B-46, and Figure B-47
in Appendix B.4.

National Transmission Planning Study

28

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

efforts toward technological standardization and institutional coordination relative to
existing practices would be necessary.
The following sections explore the diversity of network portfolios and resulting key
messages identified for the three scenarios derived from the application of the Z2N
methods and sets of tools described in Section 2. These takeaways are organized into
disaggregation (demand, generation, storage), transmission expansion and system
operations (focused on transmission) to underscore how transmission growth enables
multiple pathways and can support regionally specific solutions for transformative
futures.
Table 6. Summary of Common Themes Across Nodal Scenarios
Common Themes Across Nodal Scenarios
(Limited, AC, MT-HVDC)
-

Reinforcement (single-circuit to double-circuit) and/or reconductoring of existing transmission lines occurs
in most regions.

-

Increased meshing of existing networks improves contingency performance and collects large amounts of
renewable energy from remote parts of existing networks.

-

Development of intraregional transmission networks primarily uses existing voltage levels but with stateof-the-art high-capacity tower and conductor configurations. 37

-

Reconductoring is a possible solution for some areas where single-circuit to double-circuit expansions are
undertaken and part of the transmission network is reaching end of life and will need to undergo
modernization.

-

Significant amounts of new renewable energy and storage in parts of the country where there is little to no
HV/EHV transmission network infrastructure creates conditions where specific network topologies to
collect resources at HV levels could be a good solution and were implemented in these transmission
portfolios.

-

Although explicit transmission technology choices and expansion constraints define each transmission
portfolio, if further HVDC is implemented at scale, AC and HVDC transmission networks will need to
coexist. This results in additional needs for coordination of flows between new AC and HVDC corridors,
embedded HVDC corridors, and MT as well as meshed HVDC networks.

37
With the implicit interregional focus of the NTP Study (characterized by the need for high-capacity
interregional transfers across long distances), the use of high-capacity towers and conductor
configurations was an exogenous decision in the modeling after confirming with technical stakeholders
this was a valid path forward. It is possible in some corridors these configurations are not necessary or
feasible.

National Transmission Planning Study

29

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Table 7. Summary of Differentiated Themes for Each Nodal Scenario
Differentiated Themes Across Nodal Scenarios
Limited
-

Substantial amounts of HVAC
transmission are expanded
within transmission planning
regions to enable integration
of new VRE capacity usually
located far from main load
centers.

-

Intraregional transmission
expansion using existing
voltage levels for most regions
provides sufficient enabling
transfer capacity to move
power to load centers within
regions and to adjacent
regions.

-

-

Further enabling intraregional
expansion in some regions
requires the introduction of
new EHV voltage levels
(voltage overlays), i.e., mostly
shifting from 230 kV to 345 kV
and 345 kV to 500 kV
(minimal new voltage overlays
to 765 kV).
Reinforcement and/or
reconductoring of existing
transmission lines can be a
proxy for single-circuit to
double-circuit expansions in
specific areas (particularly
where transmission networks
are older).

AC
- Definitive need for substantial
amounts of new high-capacity,
long-distance, EHV
transmission for further
connecting transmission
planning regions.
- Further expanding existing
765-kV AC networks (relative
to 500-kV interregional
expansions).
- Contingency performance
when further expanding 765kV networks (as a large
contingency) is important when
designing transmission
expansion portfolios, i.e., need
for supporting existing and
potentially new 230-kV, 345kV, and 500-kV networks
under contingency conditions.
- In areas where 230-kV and/or
345-kV networks form the
transmission grid, highcapacity 500-kV AC
transmission seems a good
option for expansion. Singlecircuit to double-circuit or
increased network meshing at
existing voltage levels does
not prove sufficient.
-

Considering the volumes of
interregional flows, increased
coordination between regions
(for generation dispatching
needs) is expected to operate
future systems in the most
economical manner while
maintaining system reliability.

MT-HVDC
-

HVDC expansion portfolios
establish the starting points for MT
and potentially meshed HVDC
networks. Hence, the significant
expansion of high-capacity, longdistance interregional HVDC
transmission is based on bipolar,
multiterminal/meshed-ready HVDC
technologies.

-

Expanded HVDC performs the role
of bulk interregional power
transfers whereas HV and EHV
embedded AC transmission
(existing and in some cases
expanded) fulfills a supplementary
role in interregional transfers while
simultaneously supporting
contingency performance.

-

Large amounts of intraregional
HVAC networks are expanded (with
similar voltage levels as currently
being used/planned) to enable
infeed to and from HVDC converter
stations.

-

High coordination levels between
regions and across interconnects is
expected to operate future systems
in the most economical manner (for
generation dispatch and HVDC
dispatch) while maintaining system
reliability.

- Series compensation of new
EHV capacity for particularly
long distances (e.g., Western
Interconnection) is implicit in
the transmission expansion
solutions. However, this was
not explored in detail and
would need to be further
investigated considering large
amounts of inverter-based
resources.

National Transmission Planning Study

30

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

⁠Across the nodal scenarios, the expansion of transmission is always substantial. The
Limited scenario has the least amount of expansion in all metrics—total circuit-miles,
thermal capacity, and the combination of these calculated as terawatt-miles (TW-miles)
of transmission. However, the differences between the two accelerated transmission
scenarios—AC and MT-HVDC—shifts depending on the metric considered. The circuitmiles are relatively close (~73,000 miles and ~69,800 miles, respectively) whereas
thermal capacity or TW-miles is significantly less for the MT-HVDC scenario, primarily
because in the MT-HVDC scenario there are relatively long lines that make the miles
metric closer. But the HVAC transmission buildout in the AC scenario is substantially
more in the combined capacity and distance metric because the scenario includes more
shorter-distance HVAC expansions (178 TW-miles in the AC; 80 TW-miles in the MTHVDC). Further details on these findings are provided in Appendix B.4.

3.2 Translating Zonal Scenarios to Nodal Network Scenarios
3.2.1 Scale and dispersion of new resources is unprecedented
Based on ReEDS outputs, the scale and geographic dispersion of new generation and
storage resources that must be integrated into the U.S. grid in the NTP Study nodal
scenarios is unprecedented. Figure 12 shows the total installed capacity by
interconnect 38 for the nodal scenarios; Figure 13, Figure 14, and Figure 15 show the
nodal POIs (allocation of capacity to specific substations) for the three scenarios after
the disaggregation stage of the Z2N process.
The geographical dispersion and scale of new resources are the primary drivers of
resulting transmission network expansion portfolios. The scenarios that allow for
interregional transmission expansion—AC and MT-HVDC—build more capacity and
hence have a substantially larger number of POIs and larger-capacity POIs (larger HV
network injections), especially in the middle of the country where large amounts of landbased wind and solar PV are deployed. The MT-HVDC scenario is the least-cost
electricity system plan for the three scenarios, followed by the AC and then the Limited
scenario. 39 Savings in the MT-HVDC come in the form of reduced generation capital
and storage capital costs followed by reductions in fuel costs.

Further details of the installed capacity by transmission planning region and interconnection as well as
CONUS-wide for each scenario are provided in Appendix B.4.
39
Electricity system costs of these scenarios as well as their relative differences in quantum and
composition refer to the results from the zonal scenario analysis in Chapter 2.
38

National Transmission Planning Study

31

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 12. Generation and storage capacity for final nodal scenarios

The Limited scenario exhibits the least amount of installed capacity mostly because it
includes the least amount of land-based wind and solar PV (465 GW and 453 GW,
respectively) whereas a large amount of existing and new gas-fired and coal-fired
capacity are retrofitted for coal capture and storage (CCS) operations (~182 GW and
~70 GW, respectively). As a reminder, the lower level of total installed capacity in this
scenario is driven by capital costs for retrofits and new capacity as well as fuel (and to a
lesser extent by operation and maintenance cost differences). 40 This is less evident in
the AC and MT-HVDC scenarios where existing coal and gas capacity remains online
and a smaller proportion is retrofitted for CCS operations (12–13 GW and 5–40 GW,
respectively). There is 64 GW of battery energy storage system (BESS) capacity
installed in this scenario, which is a substantial ramp-up from what is currently installed
across CONUS but is substantively less than the AC scenario.
The AC scenario with the ability to expand interregional transmission via HVAC
technologies (within the same interconnection) shows a significantly larger expansion of
land-based wind and solar PV capacity (714 GW and 730 GW, respectively) and fewer
CCS retrofits of fossil generation capacity (coal and gas). Much of the wind capacity is
developed in the central wind belt of the country—in the Midwest (MISO region), along
the Great Plains (SPP region), in parts of the Northeast and Rocky Mountains
See Chapter 2, Section 3 for further details. Maps summarizing the zonal transmission expansion for
the three nodal scenarios are presented in Chapter 2 but are also provided for reference in Appendix B.4.
40

National Transmission Planning Study

32

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

(NorthernGrid and WestConnect regions), and in west Texas (ERCOT West). For
solar PV, a large amount of capacity is concentrated in the Southeast (Southeastern
Electric Reliability Council and Southeastern Regional Transmission Planning [SERTP]
regions), parts of ERCOT, and the Desert Southwest (WestConnect region). There is
~130 GW of BESS capacity installed in the AC scenario (almost twice as in the Limited
scenario).
The MT-HVDC scenario—with the ability to expand HVAC within interconnects and
expand HVDC within interconnects and across seams—exhibits larger amounts of landbased wind (770 GW) and solar capacity (600 GW) compared to the Limited scenario
but with a shift toward more land-based wind relative to solar PV capacity. As will be
demonstrated further in the following sections, there is a distinct correlation of wind
capacity expansion with increased interregional transmission. This is driven by the clear
trade-offs between long-distance, high-capacity transmission expansion and wind
resource expansion to move power from distant locations toward load centers.
All scenarios include offshore wind capacity of 47 GW being deployed by 2035 with
most of this concentrated off the Atlantic coast (42 GW) and the remainder off the
Pacific coast (~5 GW). These scenarios are based on assumed mandated offshore wind
deployment targets set by state renewable portfolio standard policies. Offshore
transmission network design options are not the focus of the NTP Study. Instead, landbased network expansion needs are established based on POIs correlated as much as
possible with other similar efforts led by DOE (Brinkman et al. 2024).
Across the three nodal scenarios, a significant amount of new generation capacity is
built in areas with very limited existing HV/EHV transmission capacity—that is, northern
parts of SPP (Nebraska, South Dakota), MISO (Minnesota), Southern SPP (Oklahoma,
Texas panhandle), eastern parts of NorthernGrid (Montana), and in the south of
WestConnect (New Mexico). This necessitates significant development of HV/EHV
transmission infrastructure because limited interregional HV/EHV transmission exists in
these regions to enable a combination of collector networks of the many POIs (mostly
VRE sources) or direct interconnection of large POIs to bulk transmission backbones for
transfers to main load centers.

National Transmission Planning Study

33

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 13. Nodal POIs, sized by capacity, for all generation types for the Limited scenario

National Transmission Planning Study

34

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 14. Nodal POIs, sized by capacity, for all generation types for the AC scenario

National Transmission Planning Study

35

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 15. Nodal POIs, sized by capacity, for all generation types for the MT-HVDC scenario

National Transmission Planning Study

36

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

3.2.2 Intraregional transmission needs are substantial, especially when
interregional options are not available
In the Limited nodal scenario, additional transmission is 77 TW-miles by 2035, 41 which
is 1.3 times the TW-miles of the existing industry planning cases. 42 New transmission
consists of local interconnection for new resources at the bulk level as well as the
enabling transit capacity to move power across nonadjacent regions to enable the
integration of new capacity usually located farther from main load centers. Figure 16
shows the spatial mapping of the Limited scenario nodal transmission portfolio. 43
Transmission portfolios predominantly include new circuits in parallel with existing
circuits and some selected new intraregional paths via double-circuit additions to enable
a nodal solution that meets the decarbonization objectives envisioned in this scenario.
Hence, preexisting voltage levels within specific regions are generally maintained.
As shown in Figure 16, the intraregional nodal HVAC transmission needs in the Limited
scenario are still substantial. 44 Local interconnection is most prevalent for new
resources at the bulk level as well as the enabling transit capacity to move power
across nonadjacent regions to enable the integration of new capacity usually located
farther from the main load centers. These intraregional expansions are particularly
notable in the southern parts of SPP, in PJM, and in northern parts of WestConnect.

It is worth noting with a caveat that this nodal transmission portfolio expansion is higher than the
corresponding ReEDS zonal expansion by 2035 (considering 1.83 TW-Miles/year constraint).
42
Industry planning cases for this study are compiled based on planning cases for 2030-31 (Table 3).
43
Table B-21, Table B-22, and Table B-23 in Appendix B.4 provide further detailed findings of the nodal
transmission solutions for this scenario.
44
A few instances of interregional transmission strengthening were required in the Limited scenario. This
is for many reasons, such as the geographic zonal designations in ReEDS and the placement of new
generation capacity (mostly wind or solar) in locations that are geographically in one zone but, when
mapped to a nodal network model, could be closer to another zone and therefore cause congestion as a
result of network impedances being represented (and not as a zonal transportation model as in ReEDS).
In general, the intention of the ReEDS Limited scenario was maintained with very few interregional
expansions.
41

National Transmission Planning Study

37

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 16. Nodal transmission expansion solution for the Limited scenario for model year 2035

National Transmission Planning Study

38

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

3.2.3 Achieving high levels of interregional power exchanges using AC
transmission technologies requires long-distance, high-capacity HV
corridors combined with intraregional reinforcements
In the AC nodal scenario, where transmission infrastructure increased to 179 TW-miles
(by 1.7 times the TW-miles by 2035 from the 2030/2031 industry planning cases),
regional diversity of resources plays an important role when designing nodal
transmission expansion portfolios. Figure 17 shows the spatial mapping of the AC nodal
transmission expansion developed for the NTP Study. 45
Transmission systems in much of the Southeast have a strong 500-kV backbone and
underlying 230-kV networks (particularly in the eastern parts of the Southeast). These
are used for the integration of predominantly new VRE generation capacity, including in
the Carolinas and Georgia and stretching into Florida. Clusters also exist in Southern
and Central Alabama as well as Tennessee. In the AC scenario, the expansion of the
500-kV network into the Midwest and farther west into the Plains enables the movement
of large volumes of power across several interfaces north-south and west-east to load
centers. This is enabled by existing 345-kV networks in the Midwest and Plains (MISO;
SPP) that are further strengthened and link with the 500-kV networks in the Southeast.
In the Northeast, solutions for expansion are selected increases in transfer capacity on
existing 345-kV networks. In PJM, there is strengthening of the existing 765-kV
networks as well as creating further 500-kV links to the Southeast and along the
Eastern Seaboard (increasing the capabilities for large north-south transfers across
longer distances).
In the Western Interconnection, 500-kV overlays and new transfer paths at 500 kV in
NorthernGrid and WestConnect help carry resources across long distances. Northern
California Independent System Operator (CAISO) solutions include further 500 kV for
interregional exchanges with NorthernGrid (Oregon) and to integrate West Coast
offshore wind capacity.
In ERCOT, the existing 345-kV networks are strengthened by creating double-circuit
345-kV paths in existing single-circuit paths while creating new paths to bring wind and
increased amounts of solar from western and northern Texas toward load centers in
Dallas-Fort Worth, Austin, and Houston. There is not a strong case for new voltage
overlays (toward 500 kV), and HVDC is precluded from this scenario as an expansion
option.
When expanding interregionally, each region of the CONUS grid exhibits characteristics
that influence network expansions. Significant amounts of new wind, solar PV, and
storage in parts of the country where there is little to no HV/EHV transmission network
infrastructure available opens the possibility for the design of specific network
topologies to connect large concentrations of resources at HV levels to integrate into
long-distance transmission corridors. This is a departure from the incremental
Table B-23 in Appendix B.4 provides further detailed findings of the nodal transmission solutions for
this scenario.
45

National Transmission Planning Study

39

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

interconnection of relatively smaller VRE plants at subtransmission voltage levels
(particularly solar PV and less for land-based wind). Areas where this is pertinent from
the findings of the NTP Study are the far northern and far southern parts of SPP,
WestConnect, parts of northern MISO, Florida Reliability Coordinating Council (FRCC),
and western and northern parts of ERCOT.
The scale of interregional power exchanges that form part of the AC scenario requires
some regions to develop high-capacity transmission infrastructure that enables transfer
flows. More specifically, regions such as MISO act as a transfer zone between SPP and
PJM whereas, similarly, the Southeast and PJM act as enablers of transfer flows for
FRCC and the Northeast (Independent System Operator of New England [ISONE]/New
York Independent System Operator [NYISO]), respectively.
The portfolios this section describes are a single implementation of interregional
transmission. Hence, it is feasible a range of alternative transmission portfolios may
provide similar technical performance where intraregional transmission needs could be
further identified and refined and region-specific planning expertise could add value to
the portfolios presented.

National Transmission Planning Study

40

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 17. Nodal transmission expansion solution for the AC Scenario for the model year 2035

National Transmission Planning Study

41

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

3.2.4 HVDC transmission buildout represents a paradigm shift and includes the
adoption of technologies currently not widespread in the United States
The 2035 MT-HVDC scenario transmission design results in 128 TW-miles of additional
transmission (a 1.5 times increase in TW-miles from the 2030/31 industry planning
cases). This includes 48 TW-miles of HVDC and 80 TW-miles of HVAC. The total
thermal capacity of HVDC expansion by 2035 is 156 GW. 46 The spatial mapping of the
transmission solution is shown in Figure 18. 47
The HVDC expansion enables substantial ties between the three U.S. interconnections,
which currently have about 2.1 GW of transfer capacity (1.3 GW between the Eastern
Interconnection and Western Interconnection and about 0.8 GW between the Eastern
Interconnection and ERCOT). Seam-crossing capacity is 46 GW, deployed through
12- x 4-GW bipoles and a 1- x 2-GW monopole. Most of this is between the Eastern
Interconnection and Western Interconnection (22 GW), further interconnecting
WestConnect and SPP as well as NorthernGrid and SPP. The Eastern Interconnection
and ERCOT capacity increases to 20 GW whereas the Western Interconnection and
ERCOT are connected via two HVDC links totaling 8 GW. The expanded ties between
ERCOT and the Western Interconnection are particularly highly used (~91%) and
unidirectional (power flowing from ERCOT to the Western Interconnection). The
expanded ties between Western Interconnection and Eastern Interconnection are
bidirectional and have a utilization of ~40%.
Within the Western Interconnection, the beginnings of an MT and meshed HVDC
network is realized at a nodal level via 20 GW of HVDC capacity (5 by 4 GW, excluding
seam-crossing capacity), connecting large amounts of wind and solar PV in the
southern parts of WestConnect to the northern parts of WestConnect and CAISO.
In the Eastern Interconnection, 90 GW of HVDC are built (18 by 4 GW and 1 by 2 GW).
These HVDC links enable power to be moved from the predominantly wind-rich areas of
northern SPP and MISO eastward, with the confluence in the central parts of MISO and
PJM. The shifting of power between southern parts of SPP, southern MISO, and the
Southeast is enabled via 16 GW (4 by 4 GW) of MT long-distance HVDC links where
8 GW of HVDC continues toward the FRCC region.
The development of the interregional HVDC transmission overlay that comprises the
MT-HVDC scenario also requires significant development of intraregional HVAC
networks to enable power infeed to and from HVDC converter stations. These can be
seen in the map in Figure 18 where HVAC networks around HVDC terminals are
expanded or increasingly meshed or, in some cases, new voltage overlays are
developed to support the scale of infeed and/or export from HVDC converter stations.
Hence, most of the HVAC expansion is via voltage levels already present in each region
46
For the HVDC buildout process, the study team used discrete building blocks: 2 GW monopole or 4 GW
bipole. These configurations were used as needed and terminated into strong areas of AC networks.
Appendix A.5 provides further details on the HVDC transmission expansion philosophy.
47
Table B-23 in Appendix B.4 provides further detailed findings of the nodal transmission solutions for this
scenario.

National Transmission Planning Study

42

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

but in some cases requires new voltage overlays. Examples of a large AC expansion is
in WestConnect North, where the existing relatively weak 230-kV and 345-kV networks
need strengthening. There are also significant changes to the underlying AC network in
southern SPP, where a 500-kV expansion enables additional movement of power from
SPP toward southern MISO.
The extent of HVDC transmission expansion in the MT-HVDC scenario combined with
the underlying nodal AC transmission expansion has not yet been seen in industry
plans in the United States (Johannes P. Pfeifenberger et al. 2023; Bloom, Azar, et al.
2021; Brown and Botterud 2021; Eric Larson et al. 2021; Christopher T. M. Clack et al.
2020; Bloom, Novacheck, et al. 2021) or internationally (European Network of
Transmission System Operators for Electricity [ENTSO-E], n.d.; Terna spa, n.d.;
National Grid ESO, n.d.; DNV 2024; CIGRE 2019, 775; MEd-TSO 2022; Empresa de
Pesquisa Energética [EPE] 2023). Similarly, considering the level of buildout envisioned
in the MT-HVDC scenario, features that could ease HVDC growth over the coming
decades include the implementation of MT and meshed-ready technology deployments;
common design principles and standardized HVDC voltage levels, communications
protocols and technology configurations would also be helpful. These were generally
assumed to exist in the MT-HVDC nodal scenario. Finally, the operational and
institutional coordination necessary in MT-HVDC production cost modeling is beyond
what is currently practiced between regions in the United States and may require new
approaches to operations via increased coordination at various operational
timescales. 48

Ongoing efforts in the United States and Europe are moving toward standardization of HVDC
transmission design and related interoperability (among others). Examples of these include DOE (2023)
in the United States and InterOPERA (InterOPERA 2023) in Europe.
48

National Transmission Planning Study

43

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Differences between MT-HVDC nodal scenario and zonal scenario
As noted in the methodology sections of this chapter, ReEDS results do not provide
prescriptive zone-to-zone discrete HVDC lines and converter pairs. When expanding
the nodal transmission, the study team made decisions about transmission expansion
using ReEDS results as a guideline, in addition to using the nodal industry planning
cases network topology, semibounded power flows after disaggregation of generation
and storage, and related information that might impact actual network expansion.
For the MT-HVDC scenario, there are many additional challenges beyond those in
the Limited and AC scenario in deciding on expansion, including the need to embed
HVDC into large and complex HVAC networks where there is limited existing HVDC
transmission. Therefore, the study team approached this by using discrete HVDC
expansions as a starting point and building out HVAC around this through an
iterative process to enable increased use of existing networks. In some cases,
express HVDC corridors skipped over ReEDS zones where power was evidently
moving through zones instead of using that zone as a source or sink. In practice,
these zones could potentially tie into the HVDC line that is passing through via
additional converter capacity. In the end, if enough transmission was added to meet
the requirements of a reliable system for 2035—balancing supply and demand and
enabling generation to move to load centers and meet demand as planned by the
ReEDS scenarios—the transmission expansion was considered complete.
The result of this approach is the MT-HVDC nodal scenario exhibits much less new
HVDC transmission capacity than seen in the zonal ReEDS scenarios. One of the
primary reasons for this is the nodal production cost modeling scenarios are not
directly assessing resource adequacy with multiyear weather data (they are run for
only one weather year). So, where ReEDS may have seen a substantial amount of
resource adequacy benefit over the full 7-year perspective (see Chapter 2), discrete
nodal production-cost and power flow models might not capture these trends and
potential additional value of transmission. Future work could benefit from verifying
network expansions on many future model years with varied weather and further
extending the Z2N workflows to additional modeling domains where this value can
be captured.
An additional consideration in building out the MT-HVDC scenario is HVDC is applied
as a transmission network expansion solution only in cases where it is a clear and
obvious solution for long-distance, high-capacity power transfer needs as guided by
the ReEDS findings. Therefore, the study team relied on it only when it was clearly the
best solution relative to more incremental AC transmission expansion.
For context on the scale of expansion achieved in the MT-HVDC scenario, global
HVDC installed capacity by the end of 2023 was ~300 GW (the 10-year pipeline is an
additional ~150 GW) (Johannes P. Pfeifenberger et al. 2023). With the full
implementation of the nodal MT-HVDC scenario for CONUS alone, the global HVDC
market would increase by 50% by 2035.

National Transmission Planning Study

44

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 18. Transmission portfolio solution for MT-HVDC scenario for the model year 2035

National Transmission Planning Study

45

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

3.3 Operations of Highly Decarbonized Power Systems
The following presents results of the annual production cost modeling for the final
contiguous U.S. nodal scenarios for the model year 2035.
3.3.1 Interregional transmission is highly used to move renewable power to load
centers but also to balance resources across regions
Transmission expansion in the AC and MT-HVDC scenarios enables large amounts of
power to flow across major interregional interfaces. As seen in the flow duration curves
in Figure 19, the scale of power flows between the Southeast and MISO (Central)
regions increases in the AC scenario—up to 10.1 GW and ~27 TWh of annual energy—
compared to the Limited scenario where only up to 2.6 GW of flow and ~10 TWh of
annual energy is exchanged across the interface. Though the dominant contributor to
the differences is the magnitude of the interregional transfer capacity, some of this
difference is also attributable to the 3%–8% of congested hours where more congestion
exists in the Limited scenario relative to the AC scenario (shown by the flat portion of
the curves). Further, the predominant flow of power from MISO (Central) to Southeast
(70%–75% of the time) indicates low-cost predominantly wind resources in the MISO
region (in all scenarios) are consumed in the Southeast for parts of the year. MISO is
also a net exporter to PJM, sending a net of 105 TWh annually.

Figure 19. Flow duration curve between MISO-Central and the Southeast
Positive values indicate power is flowing from MISO-Central to the Southeast; negative values indicate flow from Southeast to
MISO-Central.

In other areas, such as the ties between the Southeast and FRCC, the interregional
transmission infrastructure experiences increased average utilization compared to other
National Transmission Planning Study

46

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

interfaces. The lines that comprise the interface between SERTP and FRCC regions
play an important role in leveraging diurnal complementarities between solar PV
systems and BESS (in FRCC) and combined wind, solar PV, and other dispatchable
technologies (in the Southeast). This balancing of resources between regions is
demonstrated through the FRCC and Southeast flow duration curves and diurnal flow
distributions shown in Figure 20. Although there are larger maximum amounts of power
flows between these two regions in the AC relative to Limited scenario, and further
larger transfers for the MT-HVDC scenario, there is a difference in the relative
distribution of these flows as well as directionality. Figure 22 (b–d) illustrates these
differences, showing how large amounts of wind are imported into FRCC during the
early mornings and late evenings and solar is exported into the Southeast in the
afternoon, using the full capacity of the interface in both directions. This is driven by the
increased amount of solar PV in FRCC in the AC scenario. The diurnal flow in the
Limited scenario is characterized by a wider distribution of imports and exports but is
punctuated by more exports (or relatively lower imports) from FRCC to the Southeast in
the morning hours (06h00–10h00) and later evening hours (19h00–21h00), which is
particularly pronounced during the spring and summer months (March–August). The
annual average dispatch stacks in 2035 are shown in Figure 21 for FRCC and the
Southeast (for AC and MT-HVDC scenarios), demonstrating shared resources between
these two regions.

National Transmission Planning Study

47

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

(a)

(b)

(c)

(d)

Figure 20. Flow duration curves (a) and distribution of flows (b), (c), (d) between FRCC and the Southeast

National Transmission Planning Study

48

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

(a) AC scenario

(b) MT-HVDC scenario
Figure 21. Dispatch stack for peak demand period in FRCC and Southeast for (a) AC and (b) MT-HVDC

3.3.2 Diurnal and seasonal variability may require increased flexibility as well as
interregional coordination to minimize curtailment
The Limited scenario exhibits the least amount of curtailment of VRE resources relative
to the AC and MT-HVDC scenarios, driven by the combination of more installed wind
and solar capacity in the AC and MT-HVDC scenarios (with the same demand) and less
interregional transmission. As an example of this, for 15%–22% of the year, there is
more energy available from VRE sources than there is demand in the Eastern
Interconnection (greater than 1.0 in Figure 22). Consequently, even with no network
congestion, oversupply would result in VRE curtailment because there is not enough
storage capacity to consume the extra power in all periods (a trade-off established in
the zonal ReEDS scenarios between storage investment and operational costs relative
to curtailed energy).

National Transmission Planning Study

49

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 22. VRE production duration curve (normalized to demand) for the Eastern Interconnection

Figure 23 shows the monthly generation per interconnection for the AC scenario; Figure 24
demonstrates this for CONUS for the MT-HVDC scenario (monthly curtailment can be
seen in the light-gray shade). The Limited scenario exhibits monthly curtailment patterns
similar to those of the AC scenario but with lower absolute levels (as also indicated
in Figure 22). With the large amounts of VRE resources added in scenarios with
interregional transmission expansion, curtailment becomes common throughout the
year and is particularly higher in regions with disproportionately larger amounts of VRE
resources added relative to demand (SPP and MISO in particular). Similarly, the
seasonal pattern of curtailment is driven by relatively high wind energy resources in the
spring and fall, coinciding with relatively lower demand months (at an interconnection
level and CONUS level) combined with lower correlation with the availability of hydro
resources throughout other parts of the year.

National Transmission Planning Study

50

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Eastern Interconnection

Western Interconnection

ERCOT

Figure 23. Monthly generation (per interconnection) for the AC scenario
These demonstrate monthly curtailment patterns (higher curtailment in late winter and early spring); similar trends exist for the
Limited and MT-HVDC scenarios.

National Transmission Planning Study

51

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

(a)

(b)

(c)

Figure 24. Monthly generation (CONUS) for (a) MT-HVDC scenario and (b) within SPP and (c) MISO
Demonstration of monthly curtailment patterns CONUS-wide as well as specific regions with large amounts of installed land-based
wind capacity where general trends are highlighted further.

Dispatch trends demonstrating flexibility needs across the nodal scenarios are
demonstrated via dispatch stacks for various seasonal periods for the Limited, AC, and
MT-HVDC scenarios in Figure 25, Figure 26, and Figure 27, respectively. As expected,
higher levels of curtailment exist in the daytime hours when wind and solar PV are both
producing simultaneously and in scenarios where more VRE make up the resource mix
(AC and MT-HVDC). This curtailment supports balancing compared to the Limited
scenario because of the significantly more wind and solar installed for the same
demand even though there is more interregional transmission expansion. The use of
short-duration storage resources (in the form of BESS) also plays a large role in
balancing, where discharging of storage occurs in some morning hours as demand
increases but is mostly concentrated in evening hours after being charged during the
day when there is excess wind and solar. This finding is robust across interconnections
and scenarios.
The curtailment of VRE is driven primarily by the trade-offs between resources and
transmission in ReEDS. Once translated into nodal production-cost models with
National Transmission Planning Study

52

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

improved temporal and transmission networks representation, curtailment emanates
further from the combined flexibility characteristics 49 of the complementary fleet of
dispatchable technologies and interregional transmission expansion. The fast-ramping
nature and low absolute levels of residual demand after wind and solar generators are
dispatched are particularly noticeable for BESS, combined cycle gas, and combustion
engines where increased unit starts/stops, increased periods of operation at minimumstable levels, and increased unit ramping are required.

Flexibility is required for increased ramp rates, lower minimum operating levels, and more startups and
shutdowns of supply resources to meet residual demand needs (within and across regions).
49

National Transmission Planning Study

53

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Eastern Interconnection

Western Interconnection

ERCOT

Figure 25. Seasonal dispatch stacks (per interconnect) for the Limited scenario

National Transmission Planning Study

54

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Eastern Interconnection

Western Interconnection

ERCOT

Figure 26. Seasonal dispatch stacks (per interconnect) for the AC scenario

National Transmission Planning Study

55

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 27. Seasonal dispatch stacks (continental United States) for the MT-HVDC scenario

3.3.3 Regions with high amounts of VRE relative to demand become major
exporters and often exhibit very low amounts of synchronous generation
The AC and MT-HVDC scenarios represent futures with large amounts of energy
transfers between regions as transmission expansion better connects them. This could
drastically change the role of certain regions or increasingly solidify their role as large
power exporters. For example, as demonstrated for the AC scenario in Figure 28 for a
peak demand week in SPP and MISO, large amounts of VRE resources relative to
demand exist. This results in large exports of wind and solar (where supply is greater
than demand in Figure 28). 50 These exports to other regions result in operating periods
with very few online synchronous generating units in the region and large numbers of
inverter-based resources (IBRs) in the form of wind and solar generators in operation.
These operating periods are important to scope and dimension the scale of mitigation
measures and solutions to address stability concerns with such high levels of IBRs and
low levels of synchronous generation. However, these mitigation measures and
solutions are out of scope for the NTP Study.

50

Similar patterns with respect to exports of wind and solar also exist in the MT-HVDC scenario.

National Transmission Planning Study

56

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

SPP

MISO

Figure 28. Dispatch stacks of peak demand for SPP (top) and MISO (bottom) for the AC scenario

Figure 29 shows the net interchange relative to native demand for the regions across
the continental United States, where a positive indicates exports and a negative
indicates imports. Several regions move power in one predominant direction as exports
(SPP, MISO, WestConnect, ERCOT) and as imports (CAISO, PJM, NYISO) whereas
others move power bidirectionally (NorthernGrid, SERTP, FRCC, ISONE). In the
scenarios with more interregional transmission (AC and MT-HVDC), the relative amount
of demand met by imports or the amount of power exported from a region increases,
highlighting coordination between regions would be expected to increase. Across
almost all regions, the AC and MT-HVDC scenarios lead to more overall energy
exchange. More specifically,19% of the total energy consumed in the Limited scenario
flows over interregional transmission lines whereas that number increases to 28% in the
AC and 30% in the MT-HVDC scenario.

National Transmission Planning Study

57

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 29. Ratio of Net interchange to for the 11 regions all hours of the year (2035) for Limited, AC, and
MT-HVDC nodal scenarios
Positive values indicate net exports and negative values indicate net imports. The MT-HVDC scenario refers to the nodal 2035
translation of the MT scenario, which encompasses characteristics of both HVDC scenarios from the zonal scenarios.

National Transmission Planning Study

58

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

4 Western Interconnection Results for
Downstream Modeling
This section presents results derived from the earlier ReEDS scenarios; results were
informed by the baseline analysis performed on the Western Interconnection. 51 The
three earlier scenarios that are translated to nodal scenarios are all 90% by 2035
emission constrained, high-demand, with the Limited, AC, and MT-HVDC transmission
frameworks. 52 The resulting nodal transmission expansions and production cost
modeling results for the model year 2035 for the Western Interconnection are presented
in this section.
The results are structured into three categories. Infrastructure changes, the results of
the disaggregation, and transmission expansion are described in Section 4.1.
Operational results from production cost modeling simulations of the 2035 model year
follow in Section 4.2. Finally, the impact of infrastructure and operation changes are
combined in an economic evaluation and analysis of the nodal scenarios in Section 4.3.
In addition, this section uses regions familiar to stakeholders in the Western
Interconnection. See Appendix B.2, Figure B-44 for a map of these regions. 53

4.1 Translating Zonal Scenarios to Nodal Network Scenarios
This section describes the results of the resource disaggregation in the nodal models
and the transmission expansion decisions made to accommodate the new resources.
Details on the MT-HVDC design are provided in Appendix A.10.
4.1.1 Increased transmission expansion in the West via long-distance highcapacity lines could enable lower overall generation capacity investment
by connecting the resources far from load centers
Though solar PV makes up the largest share of generation capacity in all three
scenarios, the increased share of wind in the AC and MT-HVDC scenarios—enabled by
more interregional transmission capacity—makes it possible to support the same load in
the Western Interconnection with a lower level of total installed capacity. 54 Figure 30
shows the total Western Interconnection generation and storage capacity, and Figure
31 shows the resulting transmission expansion for the three scenarios.
The Limited scenario is characterized by significant amounts of solar and storage
additions, particularly in the southwest and California. These drive a substantial amount
For the baseline analysis, see Konstantinos Oikonomou et al. (2024). For details on the different
assumptions used in ReEDS to create earlier and final scenarios, see Chapter 1 of this report.
52
The MT-HVDC scenario is modeled with the Western Interconnection and the Eastern Interconnection;
however, only results from the Western Interconnection footprint are included in this section.
Appendix B.5 includes a map of the Eastern and Western Interconnection portfolios.
53
Regions referenced in Section 4 are Northwest U.S. West and East (NWUS-W, NWUS-E), Basin
(BASN), California North and South (CALN, CALS), Desert Southwest (DSW), and Rocky Mountain
(ROCK).
54
The economic analysis, Section 4.4, evaluates the trade-off between generation and transmission
capacity.
51

National Transmission Planning Study

59

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

of intraregional transmission along the southern border as well as around the Los
Angeles (LA) Basin to collect the resources and connect them to load centers.
The AC scenario, capitalizing on the ability to build longer-distance interregional AC
transmission, incorporates more high-quality wind resources from regions on the
eastern part of the Western Interconnection (ROCK, BASN, DSW) compared to the
Limited scenario. The transmission expansion helps connect the new wind resources to
load centers in Colorado, Arizona, and the coast. The transmission expansion connects
the remote wind areas to the 500-kV backbone but also reinforces the system and
expands interregional connections to allow access to these load centers. In contrast,
there is significantly less solar PV and storage capacity, particularly in the southwest in
the AC scenario compared to the Limited scenario. Accordingly, there is less
intraregional transmission, most notably along the southern border.
The MT-HVDC scenario is substantially different from the Limited and AC scenarios
because four HVDC converters with 20 GW of capacity between the western and
eastern interconnections are added. This enables a large amount of energy exchange
across the seams, with imports from the SPP region being predominantly wind. The
wind capacity in the MT-HVDC scenario is distributed similarly to the AC scenario at
higher-quality locations along the east, supported by an HVDC backbone along the
eastern part of the interconnection with further connections toward the coastal load
centers. The total wind capacity, however, is only 9 GW more than the Limited scenario
given the ability to import additional high-quality resources across the seams. 55 Like the
AC scenario, the MT-HVDC scenario has less solar and storage installed than the
Limited scenario and therefore less intraregional transmission expansion along the
southern border.
Transmission expansion decisions are categorized into three groups that are outlined
further in the following paragraphs.
The first is high-voltage transmission lines that can span large, potentially interregional,
distances but whose primary role is to collect VRE injections from remote locations and
give them a path to the larger bulk system. For example, this type of expansion
characterizes much of the built transmission in Montana or New Mexico in all scenarios.
A second type of expansion reinforces congested corridors in the existing bulk system
or expands the bulk system via new corridors. The difference between this category and
the first is the lines in this category are primarily intended to connect regions. Though
not a requirement, flow on these lines is more likely bidirectional. The 500-kV expansion
across northern Arizona is one such example, as well as the Garrison to Midpoint lines
in the Limited and AC scenarios. Perhaps most prominently, all the HVDC expansion in
the MT-HVDC scenario is in this category.

In fact, the difference between the installed wind capacity in the AC and MT-HVDC scenarios is very
close to the roughly 20 GW of HVDC capacity across the seams.
55

National Transmission Planning Study

60

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Finally, a third type of expansion focuses on intraregional reinforcement and access to
load. The expansion around the LA or Denver load pockets are examples in all the
scenarios.
Appendix B.5 provides region-specific descriptions of the rationale for the expansion
decisions.

Figure 30. Net installed capacity after disaggregation: (a) Western Interconnection-wide; (b) by subregion

National Transmission Planning Study

61

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 31. Transmission expansion results on the Western Interconnection footprint
Nodal transmission expansion is shown for the earlier rounds ReEDS scenarios with (a) Limited, (b) AC, and (c) MT-HVDC results
shown. Expansion results for the full MT-HVDC scenario are provided in Appendix B.5 (for CONUS).

National Transmission Planning Study

62

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

4.2 Operations of Highly Decarbonized Power Systems
This section presents operational results from production cost simulations on the three
translated scenarios. These results serve two distinct purposes:
• Validation: Demonstrate an operationally viable realization of the capacity
expansion results that can be used to seed further downstream models (see
Chapter 4 for power flow and Chapter 5 for stress analysis).
• Present analysis of operation under ~90% renewable penetration and the role of
transmission and generation technologies in the operation of such futures.
Appendices A.8 and A.9 provide more detail on the simulation process and the earlier
ReEDS scenario translations.
4.2.1 Operational flexibility is achieved by a changing generation mix that
correlates with the amount of interregional transmission capacity
For the Western Interconnection scenarios, as interregional transmission increases, the
share of wind in the total energy mix increases from 270 TWh (21%) in the Limited
scenario to 346 TWh (28%) in the AC scenario and 439 TWh (36%) in the MT-HVDC
scenario. The reliance on gas generation (gas turbine [GT] and combined cycle [CC])
makes up a smaller share as interregional transmission increases from 11% in the
Limited to 10% in the AC and 5% in the MT-HVDC scenario. Figure 32 shows the
annual generation in the entire Western Interconnection and by region. Interregional
imports and exports have a large impact on certain regions in the AC and MT-HVDC
scenarios. In particular, the HVDC connections (shown as DC imports in Figure 32) to
the Eastern Interconnection in the MT-HVDC scenario have a large impact on flows
around the entire interconnection.
Storage charge and discharge patterns highlight the different sources of flexibility
between the scenarios. Figure 33 shows the average weekday dispatch in the third
quarter of the year, which in the Western Interconnection contains the peak load period.
Storage plays a significant role in meeting the evening ramp and peak, but it is more
pronounced in the Limited and AC scenarios. In the MT-HVDC scenario, the lower
storage contribution is compensated by more wind as well as imports through the HVDC
links that show a similar pattern to storage. In the AC and MT-HVDC scenarios, storage
discharge reduces significantly overnight (hours 0–5) compared to the higher levels in
the Limited scenario. In the AC scenario, the nighttime generation is picked up by more
wind whereas in the MT-HVDC scenario the HVDC imports—in addition to the higher
wind generation—help drive down the share of gas generation dispatched overnight.
Figure 34 shows the average weekday storage dispatch over the whole year, further
emphasizing the wider charge and discharge range as well as overnight discharge level
in the Limited scenario.

National Transmission Planning Study

63

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 32. Annual generation mix comparison for (a) Western Interconnection and (b) by subregion
Percentages are with respect to total generation within the footprint, in other words, neglecting the DC and AC imports, and
curtailment values. Storage values are for generation mode only.

National Transmission Planning Study

64

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure 33. Average weekday in the third quarter of the model year for (a) Limited, (b) AC, and (c) MTHVDC
DC imports refer to energy imported across the seam from the Eastern Interconnection. Storage values are for generation mode
only.

Discharge/Charge [MWh]

Scenario

Lim

AC

MT-HVDC

60k
40k
20k
0
−20k
−40k
−60k
0

5

10

15

20

Hour of Day
Figure 34. Average weekday storage dispatch for the Western Interconnection
Lines are the average value; shaded areas are ±1 standard deviation around the mean.

National Transmission Planning Study

65

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

4.2.2 Transmission paths connecting diverse VRE resources will experience
more bidirectional flow and diurnal patterns
Increased interregional transmission and its correlation with more geographically
dispersed VRE resources lead to more pronounced diurnal patterns on transmission
paths, largely driven by solar PV generation, as well as changes in the dominant
directionality of flow.
The upgraded transmission paths for the additional wind resources in New Mexico,
Colorado, and Wyoming toward the California load centers passes through southern
Nevada and Arizona in the AC and MT-HVDC scenarios. Figure 35 shows the flows on
the interfaces between CALS and BASN (southern Nevada) as well as DSW
(Arizona). 56 During the nighttime hours, the magnitude of the import flows to California
increases on average by around 5 GW in both the AC and MT-HVDC scenarios
compared to the Limited scenario. During the daytime hours, the significant solar
resources in CALS reduce the flow on the interfaces, resulting in a diurnal pattern that
reflects the solar PV output.

Figure 35. Flows between Basin and Southern California and between Desert Southwest and Southern
California
Panels (a) and (c) are flow duration curves. Panels (b) and (d) show average weekday plots with ±1 standard deviation shaded.

The interface between the Basin region (BASN) and the Pacific Northwest (NWUS-W)
in Figure 36 is an example of how interregional transmission impacts the dominant flow
direction between regions. In this case, the transmission capacity between the two
regions in the three scenarios is similar, but the impact of expansion elsewhere in the
Note in the AC scenario a significant portion of the new 500-kV transmission passes through southern
Nevada whereas in the MT-HVDC scenario, the HVDC corridor connects Arizona (DSW) to southern
California.
56

National Transmission Planning Study

66

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Western Interconnection impacts how that transmission capacity is used. Central to the
shift in usage pattern is the reduction of wind resources in NWUS-W (cf. Figure 30) in
favor of eastern regions of the Western Interconnection, which is enabled by more
interregional expansion. Figure 36 shows how NWUS-W shifts from exporting power to
BASN for more than 50% of the year in the Limited scenario to importing power for 60%
(MT-HVDC) and 75% (AC) of the year. This underscores the shift in total yearly energy
exchange for NWUS-W, seen in Figure 32, from 13 TWh exporting to 7 TWh importing
and 28 TWh importing in the Limited, AC, and MT-HVDC scenarios, respectively.
In the MT-HVDC scenario, there is an HVDC path to the Pacific Northwest, which brings
wind from Montana and North Dakota and acts as an alternative wind import option for
NWUS-W. There is also an MT-HVDC link from BASN to the east that offers an
alternative export destination to NWUS-W. As a result, the BASN-NWUS-W interface
has a flatter average flow but a wider standard deviation in Figure 36, driven by the
increased variations in available flow patterns.

Figure 36. Interface between Basin and Pacific Northwest
Panel (a) flow duration curve and panel (b) average weekday shape with ±1 standard deviation shaded.

4.2.3 HVDC links between the Western and Eastern Interconnections are highly
used and exhibit geographically dependent bidirectional flows
The 20 GW of HVDC capacity added between the Western and Eastern
Interconnections in the MT-HVDC scenario is used substantially, as illustrated in the
flow duration curves of Figure 37, where flat portions represent operation at the limit. All
four HVDC links connecting the Western and Eastern Interconnections in the MT-HVDC
scenario move power in both directions—importing to the Western Interconnection and
exporting from it. The degree of bidirectionality varies geographically: larger in the north
and south and lesser in the middle between ROCK and SPP.
The northernmost connection between Montana and North Dakota exhibits a light
diurnal pattern. At times, power flows west-to-east during the day when there is much
solar PV available in the West and the link offers an alternative sink for wind. During the
evening and nighttime, the flow is predominantly east-to-west as also seen in the
storage-like behavior of HVDC for the West in Section 4.2.1. The southernmost seam,
between DSW and SPP-South, exhibits a strong diurnal pattern and is evenly split
between daily west-east and nightly east-west flows.

National Transmission Planning Study

67

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

The ROCK region, containing Colorado and parts of Wyoming, has two connections to
the Eastern Interconnection via SPP-North and SPP-South. In both, power flows
predominantly west-to-east, although east-to-west flows make up around 10% of the
time. This is likely because of ROCK’s weaker connection to the rest of the Western
Interconnection, making it easier to send power east versus west. 57

Figure 37. Flow duration curve (left) and average weekday across four Western-Eastern Interconnection
seam corridors

4.3 Economic Analysis Indicates Benefits From More Interregional
Transmission in the Considered Scenarios
This section highlights the avoided costs and other economic benefits estimated from
the earlier nodal scenarios modeled. The metrics presented are the avoided costs and
benefits to the Western Interconnection only. The results compare the three earlier
nodal scenarios modeled—Limited, AC, and MT-HVDC—and the results may not
The MT-HVDC buildout from the plains eastward is much more extensive than the buildout in the west.
Therefore, issues around weak connections on the Eastern Interconnection side are more robustly
addressed. Details of the full HVDC expansion are available in Appendix B.5.
57

National Transmission Planning Study

68

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

generalize to other transmission scenarios or portfolios. Grid planners will need to
conduct comprehensive nodal economic analysis as potential transmission projects are
planned and implemented. Because these results use earlier scenario input
assumptions, the estimates differ from the zonal economic analysis presented in
Chapter 2 of the NTP Study that uses final scenario input assumptions. The economic
results demonstrate interregional transmission could provide cost savings to the grid.
4.3.1 Increased transmission capital cost expenditures in the studied
interregional scenarios coincide with lower generation capital costs
The transmission costs for the Limited scenario total $46.2 billion including a 17.5%
adder for an AFUDC. The added transmission for the AC scenario totaled $55.1 billion.
The transmission costs for the MT-HVDC are $80.0 billion for transmission lines and
converter stations including the AFUDC. The study team estimated costs for each set of
transmission lines using the WECC Calculator (Black & Veatch 2019). Table 8, Table 9,
and Table 10 include the total number of lines, total line miles, average cost per mile,
and the total cost for each case.
Table 8. Transmission Capital Cost for the Limited Scenario

Right-of-Way Cost
Transmission Line Cost

Number of
Lines
220

Total
Mileage
10,373

Costs
($B)
0.2

220

14,905

39.1

Total Cost

46.2

Table 9. Transmission Capital Cost for AC Scenario

Right-of-Way Cost
Transmission Line Cost

Number of
Lines
232

Total
Mileage
11,971

Costs
($B)
0.2

232

18,447

46.7

Total Cost

55.1

Table 10. Transmission Capital Cost for the MT-HVDC Scenario

Right-of-Way Cost
Transmission Line Cost

Number of
Lines
225

Total
Mileage
15,164

Costs
($B)
0.2

225

24,594

53.6

Converter Station Costs

14.5

Total Cost

80.0

Table 11, Table 12, and Table 13 list the costs by voltage class to show the differences
across the scenarios considered. The Limited scenario contains the highest 230-kV
costs because of its higher reliance on intraregional expansion. The 500-kV costs
dominate the AC scenario as the primary mode of interregional expansion used. Finally,
HVDC transmission drives the MT-HVDC scenario costs.

National Transmission Planning Study

69

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Table 11. Cost by Mileage and Voltage Class for the Limited Scenario

kV
230 AC

Circuit
Count
50

Circuit
Miles
1,731

Right-ofWay Cost
($M)
20.0

Transmission
Lines ($B)
2.5

Cost/Mile
($M)
1.7

Total
Cost
($B)
3.0

345 AC

68

3,585

22.1

6.9

2.3

8.1

500 AC

185

9,590

128.8

29.8

3.7

35.1

Table 12. Cost by Mileage and Voltage Class for the AC Scenario

kV
230 AC

Circuit
Count
29

Circuit
Miles
972

Right-ofWay Cost
($M)
20.0

Transmission
Lines ($B)
1.5

Cost/Mile
($M)
2.1

Total
Cost
($B)
1.7

345 AC

54

3,844

24.7

7.4

2.6

8.7

500 AC

149

13,591

137.5

37.8

3.9

44.6

Table 13. Cost by Mileage and Voltage Class for the MT-HVDC Scenario

kV
230 AC

Circuit
Count
30

Circuit
Miles
725

Right-ofWay Cost
($M)
16.5

Transmission
Lines ($B)
1.0

Cost/Mile
($M)
1.7

Total
Cost
($B)
1.2

345 AC

92

7,667

33.2

13.3

2.0

15.7

500 AC

88

8,526

67.1

23.8

3.3

28.0

500 HVDC

15

7,676

48.7

29.7*

4.6

35.0

*Includes converter stations

ReEDS output for 2022–2034 provides the basis for the generation capital costs. 58 The
values obtained from ReEDS were in 2004$ and escalated to 2018$ using the
Consumer Price Index. The total generation capital costs are $211.3 billion for the
Limited scenario. The AC and MT-HVDC costs decrease to $163.0 billion and $151.8
billion, respectively, which is $48 billion and $59 billion less than the Limited scenario.
4.3.2 Operating costs decrease with increased interregional transmission,
resulting in greater net annualized benefits
Table 14 shows the overall annual production avoided costs. The annual operation
costs for the Limited, AC, and MT-HVDC scenarios are $9.6 billion, $7.6 billion, and
$5.5 billion, respectively. Thus, the AC scenario has avoided costs of $2.0 billion
annually whereas the MT-HVDC has avoided costs of $4.1 billion. The reduction in
fossil fuel use in the AC and MT-HVDC scenarios drives the majority of these avoided
costs. In addition to the reduction in fossil fuel usage, there is a change in the
renewable resource mix. The Limited scenario relies more on solar and storage, which
results in larger ITC payments to solar generators compared to the other scenarios.
Annualized ITC payments to solar are $2.7 billion for the Limited scenario, $2.3 billion
for the AC scenario, and $2.0 billion for the MT-HVDC scenario. The reduction in solar
is offset with an increase in wind generation in the AC and MT-HVDC scenarios, largely
in the eastern regions of the Western Interconnection. The PTC payments are $7.8
58

Generation costs are not split out by resource type.

National Transmission Planning Study

70

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

billion in the Limited scenario, $9.9 billion in the AC scenario, and $12 billion in the MTHVDC scenario.
Table 14. Summary of Annual Savings Compared to the Limited Case
AC

MT-HVDC

Annual fuel and
other operating
costs

$2.0 Billion

$4.1 Billion

Annual subsidy to
wind (PTC)

-$2.1 Billion

-$4.2 Billion

Annualized subsidy
to solar + storage
(ITC)

$0.4 Billion

$0.7 Billion

The overall net avoided costs are estimated by adding the savings from transmission
capital costs, generation capital costs, and operational costs. The study team
annualized these avoided costs to enable their addition because operational avoided
costs are expected to accrue annually whereas the capital costs are paid once and
provide infrastructure with a lifetime of many years. The study team used annualized
avoided costs instead of the net present value of avoided costs because the operating
costs are estimated from simulations of a single year (2035). The formula used to
annualize the avoided costs is as follows (Becker 2022):
𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴 𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶 = 𝐶𝐶 ×

𝑟𝑟(1 + 𝑟𝑟)𝑛𝑛
(1 + 𝑟𝑟)𝑛𝑛 − 1

Where n is the lifetime of the investment, r is the discount rate, and C is the cost being
annualized. The study team assumed the lifetime of transmission capital is 40 years and
the lifetime of generation capital is 20 years. The results are estimated using two
discount rates—3% and 5%—to show the range of avoided costs across different
discount rate assumptions. Table 15 shows the total net annualized avoided costs of the
AC scenario and MT-HVDC scenario and provides net annualized avoided costs above
the Limited scenario across the range of discount rates. The MT-HVDC scenario
delivers greater net annualized avoided costs than the AC scenario for each discount
rate used.

National Transmission Planning Study

71

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Table 15. Total Annualized Net Avoided Costs of the AC and MT-HVDC Scenarios Compared to the
Limited Scenario

Annualized Value: Transmission
and generation capital, and
operating costs
Annualized Value: Generation
capital and operating costs
(transmission capital excluded)

3%
($B)

AC

MT-HVDC

5%
($B)

3%
($B)

5%
($B)

5.0

5.5

6.6

6.9

5.4

6.0

8.1

8.9

The study team used the production cost modeling outputs to disaggregate the
estimated annual operating avoided costs to estimate benefits accruing to different
stakeholders that are part of the grid. The methodology and formulas used in this
section are described in more detail in Section 2.6. Note the benefits estimated and
identified in this section include direct economic benefits from the electricity system and
do not include other benefits such as health benefits from lowered emissions.
The difference in annual profits (revenues-costs) between the two scenarios defines
generator benefits. Note capital costs are not included as part of this annual benefit
disaggregation. Revenues for each generator are estimated as their LCOE × generation
dispatched (TWh); these revenues are shown in Table 16. This calculation assumes
LCOE is the price each generator receives under a PPA with a negotiated price at their
LCOE. 59 LCOE can be interpreted as the minimum average price a generator requires
to attract investment funding, so these results can be considered a lower bound of the
benefits to generators. In the AC and MT-HVDC scenarios, the generators function with
lower operating costs and lower revenue from power than in the Limited scenario. The
reduction in natural gas and other fossil fuel generation drives the lower operating costs.
The increase in wind generation, with a lower LCOE and thus less revenue per MWh of
power, reduces generator revenues in both the AC and MT-HVDC scenarios. However,
wind generators also receive PTC payments and, after including PTC payments to wind
generators, the benefits to generators are $1.7 billion higher than in the AC scenario
and $0.4 billion lower in the MT-HVDC compared with the Limited scenario (Table 17).

Several studies identify marginal-cost pricing in a highly decarbonized electricity system results in
prices that do not provide adequate revenue for generators to recover their costs (Milligan et al. 2017;
Blazquez et al. 2018; Pena et al. 2022).
59

National Transmission Planning Study

72

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Table 16. Detailed Generation and Revenue by Generator Type
Limited

MT-HVDC

AC

Price
Used
(LCOE)
($/MWh)

Quantity
(TWh)

Revenue
($B)

Quantity
(TWh)

Revenue
($B)

Quantity
(TWh)

Revenue
($B)

Nuclear

90.54

45

4.0

45

4.0

51

4.6

Geothermal

57.75

17

1.0

18

1.0

14

0.8

Coal

68.00

16

1.1

10

0.6

6

0.4

Wind: Landbased

22.13

270

6.0

346

7.7

439

9.7

56.87

25

1.4

25

1.4

23

1.3

Solar PV

25.43

442

11

394

9

374

9

Hydro

95.88

166

16

166

16

162

16

Natural Gas
(CC) 60

62.00

111

6.9

86

5.4

60

3.7

Natural Gas
(peaker)

157.00

35

5.6

34

5.4

6

0.9

Storage

29.32

144

4.2

118

3.5

74

2.2

Other

68.00

18

1.2

17

1.2

15

1.1

1,289

57.9

1,258

55.5

1,223

49.2

Generator
Type

Wind: Offshore

Total

LCOE is computed using the energy generation over the lifetime of the asset. This is fairly consistent
year-to-year for wind, solar, and baseload generators. For generators that are dispatchable, such as
natural gas, the capacity factor can vary more widely over time. The average LCOE estimates from
Lazard (2023) use a capacity factor of 57.5% for gas CC (operating) and 12.5% for gas peaking. The
average capacity factors from the NTP scenarios for 2035 only (coming from the PCM simulation) are
26%, 20%, and 13% for CC and 18%, 17%, and 3% for peakers for Limited, AC, and MT-HVDC,
respectively. If these capacity factors were assumed to hold for the lifetime of the generators, these gas
plants would need larger revenue to achieve full cost recovery. These larger payments would increase
generator benefits (Table 17, Line 2) by $2B in the AC and $9.1B in the MT-HVDC scenario compared to
the Limited case and reduce power purchaser benefits (Table 17, Line 6) by an equal amount. Hence, the
total benefits would be unaffected by this modeling change. Furthermore, the capacity factors are likely to
be different from their simulated 2035 values over the lifetime of the assets. Average lifetime capacity
factors would likely be higher than the 2035 snapshot because most of the fleet are not new builds and
operating during years with lower renewable penetration.
60

National Transmission Planning Study

73

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

The revenue to transmission owners is defined as the annual revenue requirements for
the transmission capital (Short, Packey, and Holt 1995). 61 The annual revenue
requirements are the minimum annual payments required to attract investment funding
and should be considered a lower bound on the transmission owners’ benefits. The AC
and MT-HVDC scenarios provide transmission owners $0.5 billion and $2.0 billion
greater revenue, respectively, than the Limited scenario. The AC and MT-HVDC
scenarios require more investment in interregional transmission and thus require higher
corresponding revenue requirements to the transmission owners to justify the
investment.
The benefit to power purchasers is defined as the reduction in payments across the
scenarios considered. The production cost model assumes load demand is equal
across the scenarios considered, so the quantity and quality of power purchased is
equal across the three scenarios. The benefit to power purchasers is the reduction in
cost required to obtain the power. Power purchaser cost is the sum of the generator
revenue and transmission owner revenue. In the AC and MT-HVDC cases, payments to
generators decrease whereas payments to transmission owners increase. In sum, the
power purchasers add a benefit of $1.9 billion and $6.7 billion annually under the AC
and MT-HVDC scenarios, respectively, compared to the Limited scenario.
The final stakeholder considered is the taxpayers. The wind generators receive PTC
payments from taxpayers. The taxpayers’ benefit is defined as the avoided cost of tax
payments. The negative benefits shown are the result of increased tax costs. The study
team assumed taxes are collected when the PTC payments are made and do not
formally model a tax system where taxes may be collected at a different time than when
the subsidies are distributed. The PTC payments increase by $2.1 billion and $4.2
billion in the AC and MT-HVDC scenarios, respectively, compared to the Limited
scenario, so the taxpayers’ benefits are -$2.1 billion and -$4.2 billion for the AC and MTHVDC scenarios.
Overall, these results show, in general, the benefits are distributed among the different
stakeholders. The generators experience a negative benefit in the MT-HVDC scenario
compared to the Limited scenario. However, renewable generation supplants
substantial fossil fuel generation and thus requires less remuneration. Though the
taxpayers earn a negative benefit in this disaggregation, they are broadly the same
collective of end consumers that will have the total benefits passed on to them through
the ratemaking process.

Annual revenue requirement can be computed using the formula 𝐴𝐴𝐴𝐴𝐴𝐴 = 𝐼𝐼 × 𝑟𝑟(1 + 𝑟𝑟)𝑛𝑛 /((1 + 𝑟𝑟)𝑛𝑛 − 1)
where I is the initial capital investment, r is the discount rate, and n is the number of annual payments
expected. The values shown use r = 5% and n = 40 years.
61

National Transmission Planning Study

74

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Table 17. Disaggregation of Annual Benefits According to Stakeholders
Stakeholder

Component

Limited
($B)

AC
($B)

Scenario

MT-HVDC
($B)

Scenario

Benefit*

Scenario

Benefit*

1. Gen cost

9.6

7.6

-2.0

5.5

-4.1

2. Gen revenue from power

57.9

55.5

-2.4

49.2

-8.7

3. Gen revenue from PTC

7.8

9.9

2.1

12

4.2

Generators

4. Total gen benefit
(2+3-1)

56.1

57.8

1.7

55.7

-0.4

Transmission
Owners

5. Transmission owner
revenue

2.7

3.2

0.5

4.7

2.0

6. Payment to generators for
power

57.9

55.5

-2.4

49.2

-8.7

7. Payment to transmission
owners

2.7

3.2

0.5

4.7

2.0

Power
Purchasers

8. Total load purchaser
benefit -(6+7)

-60.6

-58.7

1.9

-53.9

6.7

Taxpayers

9. Payment for PTC

7.8

9.9

2.1

12

4.2

10. Total taxpayer benefit (-9)

-7.8

-9.9

-2.1

-12

-4.2

-

-

2.0

-

4.1

Total (4+5+8+10)

* The benefit for the Interregional and MT-DC scenarios is defined as the difference between the benefits of that scenario and the
Limited AC scenario. Total benefits are not shown for the individual scenarios because they have practical meaning only when two
scenarios are compared.

National Transmission Planning Study

75

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

5 Conclusions
This chapter documents the nodal scenario development, which includes several
rounds of production cost modeling and DC power flow steady-state and contingency
analysis as well as select economic analysis for the NTP Study. This nodal analysis
across three scenarios (Limited, AC, and MT-HVDC) is designed to accomplish two
primary objectives: 1) verify the scenarios are feasible given network and operational
constraints and 2) provide insights about the role of transmission operations for a range
of transmission topologies.
The results of this chapter demonstrate transformative scenarios, where large
expansions of new interregional transmission significantly change the operational
relationships between transmission planning regions, can reach 90% decarbonization
by 2035, and pass preliminary reliability tests, such as select single contingencies. The
nodal transmission expansions show long-distance, high-capacity HV and EHV
transmission effectively moves power from more remote areas toward load centers,
enables bidirectional power transfers between regions, and can also play critical roles in
day-to-day balancing intraregionally and interregionally and between nonadjacent
regions. Also apparent in developing the three nodal expansions is that transmission
can adapt to different types of futures—those that have more local generation such as
the Limited scenario, and the interregional expansion scenarios (AC and MT-HVDC),
where longer-distance and higher-capacity transmission is deployed. Substantial
transmission expansion was required in all the nodal scenarios modeled for this study.
Transmission buildout scenarios using multiterminal and potentially meshed HVDC
technology represent the lowest-cost scenarios. However, the magnitude of additional
interregional transmission capacity arising from these scenarios is far advanced from
existing regional transmission planning and merchant transmission schemes. HVDC
network solutions will also require additional strengthening of intraregional AC networks.
Moreover, the HVDC scenarios present opportunities for seam-crossing HVDC
transmission between the Eastern, Western, and ERCOT interconnections. Study
results show this interregional HVDC transmission infrastructure is heavily used.
Limited interregional expansion scenarios can be operated reliably; however, they differ
from the AC and MT-HVDC scenarios by having greater local generation ramping and
balancing needs (greater need for dispatchable capacity, including clean thermal
generation, which exhibits greater variable costs considering fuel needs).
The methods developed for the transmission portfolio analysis are novel in that they
could be applied at a large-geographic scale and include both production cost modeling
and rapid DC-power-flow-informed transmission expansions. The incorporation of
capacity expansion modeling data was also innovative. This is not the first example of
closely linking capacity expansion modeling to production cost and power flow models
in industry or the research community, but several advancements were made to
realistically capture the various network points-of-interconnection for large amounts of
wind and solar, build out the local collector networks if necessary, and validate

National Transmission Planning Study

76

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

interregional solutions factor in the local transmission challenges that might arise from
scenarios that reach 90% reduction in emissions by 2035.

5.1 Opportunities for Further Research
The methods to model the U.S. electricity system with network-level detail and with
highly decarbonized systems stretch the data management and computational limits of
many methods typically used for power system analysis. In addition, this study reveals
many assumptions about the large-scale transmission expansions and power system
operations that could be strengthened with further study:
• Visualization: Modeling CONUS results in huge amounts of data to manage.
Visualization tools have helped the study team rapidly assess the success of a
transmission line development or the placement of renewable resources on the
network. But additional work on visualizing these systems should continue to be
an area of research to speed analysis and allow for easier stakeholder
interactions.
• Network formulations: Improvements in the representation of network
constraints in nodal production cost models should allow improved computational
efficiency of branch flow constraints for an increased number of HV/EHV
transmission elements.
• HVDC dispatch strategies and interregional coordination: Further implement
different strategies for the dispatch of point-to-point, embedded HVDC links and
MT or meshed HVDC networks in nodal production cost models.
• Interregional coordination: Undertake the evaluation of day-ahead and realtime interregional coordination algorithms implicitly considering information
asymmetry and uncertainty between balancing areas (plant outages, wind/solar
production, and demand).
• Power flow control devices: Improve the representation of at-scale grid
enhancing technologies (GETs) and power flow controlling device capabilities
(e.g., phase-shift transformers and static synchronous series compensators).
• Direct integration of AC power flow: Integrating AC power flow across
scenarios (see Chapter 4) by adding a stage to the transmission expansion
workflow could improve the solutions. However, more work would be needed to
develop robust model linkages where feedback from AC power flow could rapidly
inform development decisions.
• Durable contingency sets: Develop further methods to identify critical
contingency sets for various configurations of the future contiguous U.S. grid
under many contingency conditions.

National Transmission Planning Study

77

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

References

Anderlini, Luca, and Leonardo Felli. 2006. “Transaction Costs and the Robustness of the
Coase Theorem.” The Economic Journal 116 (508): 223–45.
https://doi.org/10.1111/j.1468-0297.2006.01054.x.
Becker, Denis Mike. 2022. “Getting the Valuation Formulas Right When It Comes to
Annuities.” Managerial Finance 48 (3): 470–99. https://doi.org/10.1108/MF-03-2021-0135.
Bezanson, Jeff, Alan Edelman, Stefan Karpinski, and Viral B. Shah. 2017. “Julia: A Fresh
Approach to Numerical Computing.” SIAM Review 59 (1): 65–98.
Black & Veatch. 2019. “TEPPCC Transmission Cost Calculator.”
https://www.wecc.org/Administrative/TEPPC_TransCapCostCalculator_E3_2019_Update
.xlsx.
Bloom, Aaron, Lauren Azar, Jay Caspary, Debra Lew, Nicholas Miller, Alison Silverstein,
John Simonelli, and Robert Zavadil. 2021. “Transmission Planning for 100% Clean
Electricity.” https://www.esig.energy/transmission-planning-for-100-clean-electricity/.
Bloom, Aaron, Josh Novacheck, Gregory L Brinkman, James D Mccalley, Armando L.
Figueroa-Acevedo, Ali Jahanbani Ardakani, Hussam Nosair, et al. 2021. “The Value of
Increased HVDC Capacity Between Eastern and Western U.S. Grids: The
Interconnections Seam Study.” IEEE Transactions on Power Systems.
https://doi.org/10.1109/tpwrs.2021.3115092.
Brinkman, G., Bannister, M., Bredenkamp, S., Carveth, L., Corbus, D., Green, R., Lavin,
L., Lopez, A., Marquis, M., Mowers, J., Mowers, M., Rese, L., Roberts, B., Rose, A.,
Shah, S., Sharma, P., Sun, H., Wang, B., Vyakaranam, B., ... Abdelmalak, M. 2024.
Atlantic Offshore Wind Transmission Study: U.S. Department of Energy (DOE), Energy
Efficiency & Renewable Energy (EERE). https://doi.org/10.2172/2327027.
Brown, Patrick R. and Audun Botterud. 2021. “The Value of Inter-Regional Coordination
and Transmission in Decarbonizing the US Electricity System.” Joule 5 (1): 115–34.
https://doi.org/10.1016/j.joule.2020.11.013.
Bushnell, James B. and Steven E. Stoft. 1997. “Improving Private Incentives for Electric
Grid Investment.” Resource and Energy Economics 19: 85–108.
https://doi.org/10.1016/S0928-7655(97)00003-1.
Bushnell, James and Steven Stoft. 1996. “Grid Investment: Can a Market Do the Job?”
The Electricity Journal 9 (1): 74–79. https://doi.org/10.1016/S1040-6190(96)80380-0.
Buster, Grant, Michael Rossol, Paul Pinchuk, Brandon N. Benton, Robert Spencer, Mike
Bannister, and Travis Williams. 2023. “NREL/reV: reV 0.8.0.” Zenodo.
https://doi.org/10.5281/zenodo.8247528.
California ISO (CAISO). 2017. “Transmission Economic Assessment Methodology.”
California ISO.
https://www.caiso.com/documents/transmissioneconomicassessmentmethodologynov2_2017.pdf.

National Transmission Planning Study

78

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Chowdhury, Ali and David Le. 2009. “Application of California ISO Transmission
Economic Assessment Methodology (Team) for the Sunrise Powerlink Project.” In 2009
IEEE Power & Energy Society General Meeting, 1–5. Calgary, Canada: IEEE.
https://doi.org/10.1109/PES.2009.5275185.
Christopher T. M. Clack, Aditya Choukulkar, Brianna Coté, and Sarah McKee. 2020.
“Transmission Insights from ‘ZeroByFifty.’” https://www.vibrantcleanenergy.com/wpcontent/uploads/2020/11/ESIG_VCE_11112020.pdf.
CIGRE WG B4.72. 2022. “DC Grid Benchmark Models for System Studies - Technical
Brochure: 804.” CIGRÉ. https://www.e-cigre.org/publications/detail/804-dc-gridbenchmark-models-for-system-studies.html.
CIGRE, WG C1.35. 2019. “CIGRE TB 775: Global Electricity Network - Feasibility Study.”
https://www.e-cigre.org/publications/detail/775-global-electricity-network-feasibilitystudy.html.
Conlon, Terence, Michael Waite, and Vijay Modi. 2019. “Assessing New Transmission
and Energy Storage in Achieving Increasing Renewable Generation Targets in a
Regional Grid.” Applied Energy 250 (September): 1085–98.
https://doi.org/10.1016/j.apenergy.2019.05.066.
Cramton, Peter C. 1991. “Dynamic Bargaining with Transaction Costs.” Management
Science 37 (10): 1221–33. https://doi.org/10.1287/mnsc.37.10.1221.
Dale Osborn. 2016. “ARPA-E Transmission Planning.” https://arpae.energy.gov/sites/default/files/ARPA-E%20Dale%20Osborn.pdf.
DNV. 2024. “The ASEAN Interconnector Study.” 2024.
https://www.dnv.com/publications/asean-interconnector-study/.
Doorman, Gerard L. and Dag Martin Frøystad. 2013. “The Economic Impacts of a
Submarine HVDC Interconnection between Norway and Great Britain.” Energy Policy 60
(September): 334–44. https://doi.org/10.1016/j.enpol.2013.05.041.
Duke Energy. n.d. “Rights of Way For Transmission Line.” Duke Energy. Accessed May
9, 2024. https://www.duke-energy.com/Community/Trees-and-Rights-of-Way/What-canyou-do-in-Right-of-Way/Transmission-Lines-Guidelines.
E3. 2019. “Black & Veatch Transmission Line Capital Cost Calculator.”
https://www.wecc.org/Administrative/TEPPC_TransCapCostCalculator_E3_2019_Update
.xlsx.
Egerer, Jonas, Friedrich Kunz, and von Hirschhausen, Christian. 2012. “Development
Scenarios for the North and Baltic Sea Grid - A Welfare Economic Analysis.” Deutsches
Institut für Wirtschaftsforschung.
Empresa de Pesquisa Energética (EPE). 2022. “Estudos Para A Expansao Da
Transmissao: Nota Tecnica.” https://www.epe.gov.br/sites-pt/publicacoes-dadosabertos/publicacoes/PublicacoesArquivos/publicacao-629/NT_EPE-DEE-NT-0842021.pdf.

National Transmission Planning Study

79

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

———. 2023. “Plano Decenal de Expansão de Energia 2032.” EPE. 2023.
https://www.epe.gov.br/pt/publicacoes-dados-abertos/publicacoes/plano-decenal-deexpansao-de-energia-2032.
Eric Larson, Chris Greig, Jesse Jenkins, Erin Mayfield, Andrew Pascale, Chuan Zhang,
Joshua Drossman, et al. 2021. “Net-Zero America: Potential Pathways, Infrasstructure,
and Impacts.” https://netzeroamerica.princeton.edu/.
European Network of Transmission System Operators for Electricity (ENTSO-E). n.d.
“TYNDP 2024 Project Collection.” Accessed May 9, 2024.
https://tyndp2024.entsoe.eu/projects-map.
Gerbaulet, C. and A. Weber. 2018. “When Regulators Do Not Agree: Are Merchant
Interconnectors an Option? Insights from an Analysis of Options for Network Expansion in
the Baltic Sea Region.” Energy Policy 117 (June): 228–46.
https://doi.org/10.1016/j.enpol.2018.02.016.
Gramlich, Rob. 2022. “Enabling Low-Cost Clean and Reliable Service through Better
Transmission Benefits Analysis: A Case Study of MISO’s Long Range Transmission
Planning.” American Council on Renewable Energy. https://acore.org/wpcontent/uploads/2022/08/ACORE-Enabling-Low-Cost-Clean-Energy-and-ReliableService-Through-Better-Transmission-Analysis.pdf.
Hogan, William. 1992. “Contract Networks for Electric Power Transmission.” Journal of
Regulatory Economics 4: 211–42.
Hogan, William W. 2018. “A Primer on Transmission Benefits and Cost Allocation.”
Economics of Energy & Environmental Policy 7 (1). https://doi.org/10.5547/21605890.7.1.whog.
ICF. n.d. “WECC Risk Mapping.” Accessed May 9, 2024.
https://ecosystems.azurewebsites.net/WECC/Environmental/.
InterOPERA. 2023. “InterOPERA: Enabling Multi-Vendor HVDC Grids (Grant Agreement:
No.101095874).” InterOPERA. 2023. https://interopera.eu/.
Johannes P. Pfeifenberger, Linquan Bai, Andrew Levitt, Cornelis A. Plet, and Chandra M.
Sonnathi. 2023. “THE OPERATIONAL AND MARKET BENEFITS OF HVDC TO
SYSTEM OPERATORS.” https://www.brattle.com/wp-content/uploads/2023/09/TheOperational-and-Market-Benefits-of-HVDC-to-System-Operators-Full-Report.pdf.
Joskow, Paul and Jean Tirole. 2005. “MERCHANT TRANSMISSION INVESTMENT.”
Journal of Industrial Economics 53 (2): 233–64. https://doi.org/10.1111/j.00221821.2005.00253.x.
Konstantinos Oikonomou et al. 2024. “National Transmission Planning Study: WECC
Base Case Analysis.” Richland Washington: Pacific Northwest National Laboratory.
Kristiansen, Martin, Francisco D. Muñoz, Shmuel Oren, and Magnus Korpås. 2018. “A
Mechanism for Allocating Benefits and Costs from Transmission Interconnections under
Cooperation: A Case Study of the North Sea Offshore Grid.” The Energy Journal 39 (6):
209–34. https://doi.org/10.5547/01956574.39.6.mkri.

National Transmission Planning Study

80

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Lai, Chun Sing and Malcolm D. McCulloch. 2017. “Levelized Cost of Electricity for Solar
Photovoltaic and Electrical Energy Storage.” Applied Energy 190 (March): 191–203.
https://doi.org/10.1016/j.apenergy.2016.12.153.
Lara, José Daniel, Clayton Barrows, Daniel Thom, Dheepak Krishnamurthy, and Duncan
Callaway. 2021. “PowerSystems.Jl — A Power System Data Management Package for
Large Scale Modeling.” SoftwareX 15 (July): 100747.
https://doi.org/10.1016/j.softx.2021.100747.
Lazard. 2023. “2023 Levelized Cost Of Energy+.” Https://Www.Lazard.Com. 2023.
https://www.lazard.com/research-insights/2023-levelized-cost-of-energyplus/.
Lubin, Miles, Oscar Dowson, Joaquim Dias Garcia, Joey Huchette, Benoît Legat, and
Juan Pablo Vielma. 2023. “JuMP 1.0: Recent Improvements to a Modeling Language for
Mathematical Optimization.” Mathematical Programming Computation 15: 581–89.
https://doi.org/10.1007/s12532-023-00239-3.
Maclaurin, Galen, Nicholas Grue, Anthony Lopez, Donna Heimiller, Michael Rossol,
Grant Buster, and Travis Williams. 2019. The Renewable Energy Potential (reV) Model: A
Geospatial Platform for Technical Potential and Supply Curve Modeling. Golden, CO:
National Renewable Energy Laboratory. NREL/TP-6A20-73067.
https://doi.org/10.2172/1563140.
MEd-TSO. 2022. “Masterplan of Mediterranean Interconnections.” 2022.
https://masterplan.med-tso.org/.
Mezősi, András and László Szabó. 2016. “Model Based Evaluation of Electricity Network
Investments in Central Eastern Europe.” Energy Strategy Reviews 13–14 (November):
53–66. https://doi.org/10.1016/j.esr.2016.08.001.
National Grid ESO. n.d. “Beyond 2030.” Accessed May 9, 2024.
https://www.nationalgrideso.com/future-energy/beyond-2030.
National Renewable Energy Laboratory (NREL). 2021a. “The North American Renewable
Integration Study: A U.S. Perspective.” https://www.nrel.gov/docs/fy21osti/79224.pdf.
———. 2021b. “The North American Renewable Integration Study: A U.S. Perspective.”
https://www.nrel.gov/docs/fy21osti/79224.pdf.
———. 2024. “Sienna.” Sienna. 2024. https://www.nrel.gov/analysis/sienna.html.
Neuhoff, Karsten, Rodney Boyd, and Jean-Michel Glachant. 2012. “European Electricity
Infrastructure: Planning, Regulation, and Financing.”
Nguyen, Quan, Hongyan Li, Pavel Etingov, Marcelo Elizondo, Jinxiang Zhu, and Xinda
Ke. 2024. “Benefits of Multi-Terminal HVdc under Extreme Conditions via Production
Cost Modeling Analyses.” IEEE Open Access Journal of Power and Energy, 1–1.
https://doi.org/10.1109/OAJPE.2024.3376734.
North American Electric Reliability Corporation (NERC). 2020. “TPL-001-5 —
Transmission System Planning Performance Requirements.”
https://www.nerc.com/pa/Stand/Reliability%20Standards/TPL-001-5.pdf.

National Transmission Planning Study

81

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Nowotarski, Jakub and Rafał Weron. 2018. “Recent Advances in Electricity Price
Forecasting: A Review of Probabilistic Forecasting.” Renewable and Sustainable Energy
Reviews 81 (January): 1548–68. https://doi.org/10.1016/j.rser.2017.05.234.
NREL (National Renewable Energy Laboratory). 2023. “2023 Annual Technology
Baseline.” Golden, CO. https://atb.nrel.gov/.
Olmos, Luis, Michel Rivier, and Ignacio Perez-Arriaga. 2018. “Transmission Expansion
Benefits: The Key to Redesigning the Regulation of Electricity Transmission in a Regional
Context.” Economics of Energy & Environmental Policy 7 (1).
https://doi.org/10.5547/2160-5890.7.1.lolm.
Pfeifenberger, Johannes. n.d. “Transmission Cost Allocation: Principles, Methodologies,
and Recommendations.” Brattle Group.
Pletka, Ryan, Khangura, Jagmeet, Andy Rawlins, Elizabeth Waldren, and Dan Wilson.
2014. “Capital Costs for Transmission and Substations: Updated Recommendations for
WECC Transmission Expansion Planning.” 181374. Black & Veatch.
https://efis.psc.mo.gov/Document/Display/24603.
Sanchis, Gerald, Brahim Betraoui, Thomas Anderski, Eric Peirano, Rui Pestana, Bernard
De Clercq, Gianluigi Migliavacca, Marc Czernie, and Mihai Paun. 2015. “The Corridors of
Power: A Pan-European \"Electricity Highway\" System for 2050.” IEEE Power and
Energy Magazine 13 (1): 38–51. https://doi.org/10.1109/MPE.2014.2363528.
Short, W., D. J. Packey, and T. Holt. 1995. A Manual for the Economic Evaluation of
Energy Efficiency and Renewable Energy Technologies. Golden, CO: National Renewable
Energy Laboratory. NREL/TP-462-5173, 35391. https://doi.org/10.2172/35391.
Stenclik, Derek and Deyoe, Ryan. 2022. “Multi-Value Transmission Planning for a Clean
Energy Future: A Report of the Energy Systems Integration Group’s Transmission
Benefits Valuation Task Force.” Energy Systems Integration Group.
https://www.esig.energy/multi-value-transmission-planning-report/.
Terna spa. n.d. “Grid Development Plan 2023.” Accessed May 9, 2024.
https://www.terna.it/en/electric-system/grid/national-electricity-transmission-griddevelopment-plan.
U.S. Department of Energy (DOE). 2022a. “Federal Solar Tax Credits for Businesses.”
Energy.Gov. 2022. https://www.energy.gov/eere/solar/federal-solar-tax-credits-businesses.
———. 2022b. “WINDExchange: Production Tax Credit and Investment Tax Credit for
Wind Energy.” 2022. https://windexchange.energy.gov/projects/tax-credits.
———. 2023. “WETO Releases $28 Million Funding Opportunity to Address Key
Deployment Challenges for Offshore, Land-Based, and Distributed Wind (DE-FOA0002828).” Energy.Gov. 2023. https://www.energy.gov/eere/wind/articles/weto-releases28-million-funding-opportunity-address-key-deployment-challenges.
Western Electricity Coordinating Council. 2022. “Anchor Data Set (ADS).” 2022.
https://www.wecc.org/ReliabilityModeling/Pages/AnchorDataSet.aspx.
———. n.d. “Anchor Data Set (ADS).”
https://www.wecc.org/ReliabilityModeling/Pages/AnchorDataSet.aspx.
National Transmission Planning Study

82

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Appendix A. Methodology
A.1 Nodal Datasets
Table A-18. Summary of Nodal Baseline Datasets for Contiguous United States (CONUS)
Eastern
Interconnection

Western
Interconnection

ERCOT

CONUS

Nodes

95.9

23.9

6.8

126.8

Branches (lines/cables/trafos)

115.7

28.1

8.4

152.3

Loads

41.2

14.6

3.7

59.6

Generators

10.8

4.3

0.7

15.5

Quantity (000s)

ERCOT = Electric Reliability Council of Texas

Table A-2. Overview of Data Sources Used for Building CONUS Datasets
Eastern
Interconnection

Western
Interconnection

ERCOT

MMWG 20312

WECC ADS 2030 v1.5

EnergyVisuals5

Node mapping (spatial)

NARIS

NARIS/EnergyVisuals

NARIS

Generation capacity
(technology)

NARIS

WECC ADS 2030 v1.5

NARIS

NARIS, EIA CEMS

WECC ADS 2030

NARIS

Description
Network topology (node/branch
connectivity)1

Generation techno-economic
characteristics3
Demand

EER

EER

EER

4

NARIS

WECC ADS 2030

NARIS

Variable renewable energy
(VRE) time series

reV

reV

reV

Hydro (energy constraints)

1

Augmented through stakeholder feedback to include the most recent available data on network updates/additions.

2

ERAG Multiregion Modeling Working Group (MMWG) 2031 series.

3

Includes heat rates, minimum up-/downtimes, ramp rates, and minimum stable operating levels.

4

Hourly/daily/monthly energy budgets (as appropriate).

5

Power flow case files (2021 planning cases).

ADS = Anchor Dataset (Western Electricity Coordinating Council, n.d.); CEMS = continuous emission monitoring system; EER =
Evolved Energy Research; EPA = U.S. Environmental Protection Agency; MMWG = Multiregional Modeling Working Group;
NARIS = North American Renewable Integration Study (National Renewable Energy Laboratory [NREL] 2021b); reV =
Renewable Energy Potential Model (Maclaurin et al. 2019); WECC = Western Electricity Coordinating Council

National Transmission Planning Study

83

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

A.2 Benefits and Challenges in Implementation of CONUS-Scale
Databases
The benefits of implementing CONUS-scale nodal databases are summarized next:
• Model fidelity: Increased model fidelity and insights into operations and
transmission use (beyond zonal capacity expansion)
• Discretization: Identification of discrete interzonal and intrazonal transmission
loading, congestion, and expansion needs (including potential transit needs for
nonadjacent regions)
• Data management: Enabling more seamless data flow and obtaining information
to feed forward and feed back to other modeling domains.
Appreciating these benefits, the challenges in implementation are summarized next:
• Dataset maintenance: Consistent maintenance of datasets across regions and
interconnects to ensure relevance and accuracy is challenging because each
region and interconnection applies different approaches when collating data into
aggregated datasets. Similarly, regions within each interconnection have varying
levels of alignment with interconnectionwide modeling practices.
• Dataset updates: Another challenge that has emerged is the potential time lag
to the latest available nodal information (specifically demand, generation
capacity, and network topologies), which can result in the potential for perpetual
chasing of data. This is not necessarily specific to the National Transmission
Planning Study (NTP Study) but should be considered for future similar CONUSscale work to support repeatability and potential periodic updates of CONUSscale interregional transmission planning efforts.
• Effort: The collation of datasets and model building is a data- and labor-intensive
undertaking. The NTP Study has established a structured set of workflows to
undertake this in future and address this challenge (in addition to the
abovementioned challenges of dataset maintenance and updating).
• Interregional focus: There is the potential to overly focus on regionally specific
intraregional transmission needs and solutions. This can lead to a false sense of
accuracy within regions if one deviates from the primary objectives and drivers of
the NTP Study (and other similar efforts)—that is, interregional transmission
needs and enabling intraregional transmission needs (transit needs).

National Transmission Planning Study

84

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

A.3 Disaggregation: Adding Generation Capacity to Nodal Models
The process of adding capacity to the nodal model integrates many considerations and
is implemented in an internally developed NREL tool called ReEDS-to-X (R2X):
1. Average generator sizes per technology are based on the Western Electricity
Coordinating Council (WECC) Anchor Dataset (ADS) data for standard
technology-specific generators (Western Electricity Coordinating Council 2022)
or the median generator size of the technology in the zone in which it is being
added.
2. Nodal injection limits are applied (both for number of units and capacity) to avoid
large generator injections and large numbers of generators connected to
individual nodes.
3. For variable renewable energy (VRE) technologies, which use a time series for
their characterization (fixed injections), the National Renewable Energy
Laboratory (NREL) Renewable Energy Potential Model (reV) (Buster et al. 2023)
is used in combination with a k-means clustering technique to aggregate wind
and solar photovoltaic (PV) sites to points of interconnection (POIs) (Figure 5).
4. Distributed solar PV is prioritized to nodes with high load participation factors
(LPFs), aligning with the expected distributed resource location closer to large
loads.
5. Heuristics for battery energy storage systems (BESS) are applied by co-locating
4-hour BESS to solar PV POIs and 8-hour BESS to land-based-wind POIs (with
a 50% capacity limit). Following this, any remaining BESS capacity is allocated to
nodes with high LPFs.

A.4 Disaggregation: Zonal-to-Nodal Demand
Figure A-1 demonstrates geospatially how a specific Regional Energy Deployment
System (ReEDS) zone, highlighted in (a) (Northern Colorado), is disaggregated to the
nodal loads within that zone. This is based on the established LPFs of each node for the
Northern Colorado zone (shown via the relative size of each of the markers in (b)
of Figure A-1).

National Transmission Planning Study

85

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

(a)

(b)

Figure A-1. Illustration of zonal-to nodal (Z2N) disaggregation of demand (a) CONUS and (b) zoom to
Colorado
In (a), a single ReEDS zone is highlighted; in (b), a zoom into northern Colorado illustrates individual node load through which LPFs
are calculated and zonal ReEDS load is disaggregated.

A.5 Transmission Planning Principles
The NTP Study applies several building blocks for the discrete decisions made about
the transmission planning process. Where possible, the following principles are applied
to expand both interregional and intraregional transmission:
• Right-of-way expansion: Expansion of an existing single-circuit to double-circuit
overhead line
• New corridor: Expansion of a new double-circuit overhead line
• Voltage overlay: The ability to move from established voltages to highercapacity voltage levels, e.g., 345 kilovolts (kV) to 500 kV
• High-voltage direct current (HVDC): Termination into well-interconnected
areas of existing alternating current (AC) transmission networks with converters
of 2-gigawatt (GW) monopole or 4-GW bipole at a time (further details are
provided in Appendix A.6).
The increasingly granular representation of network constraints within the staged
transmission expansion process is established through the combined monitoring or
bounding of flows across preexisting and new interfaces. The study authors collected
existing tie-line interfaces from public sources and stakeholders while defining new
interfaces between transmission planning region boundaries (based on the aggregated
subtransmission regions in Appendix B.2). In addition, interregional tie-lines were
always monitored given the interregional emphasis of the study.
The increasingly granular representation of network constraints that forms part of the
zonal-to-nodal transmission expansion workflow is shown in Figure A-2. This staged
National Transmission Planning Study

86

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

approach enables the exploration of line loading, congestion, and use to highlight
transmission expansion needs while improving model tractability in Stage 1 of the
workflow.

Figure A-2. Illustration of nodal transmission network constraint formulations

A.6 Transmission Planning Approach for HVDC Networks
The unique nature of HVDC network design (notwithstanding classical point-to-point
types of HVDC infrastructure) shifts the staged transmission planning (Figure 6)
approach slightly for a scenario with large amounts of multiterminal HVDC zonal
investments. This approach is outlined next:
1. Establish a conceptual topology: Establish an initial HVDC network focused
on the building blocks for HVDC. These are assumed as 2-GW monopole and
4-GW bipole pairs at a time. From this, establish appropriate nodes for discrete
HVDC links into the combination of large load zones, large concentrations of
generation capacity (particularly VRE), and well-interconnected areas of existing
HVAC networks with a capability to absorb/evacuate large amounts of power—
that is, areas with good network connectivity, higher voltage levels, and strong
short-circuit levels (closer to synchronous generators).
National Transmission Planning Study

87

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

2. Prioritize large interregional transfers: Although generation and storage
capacity are prescribed from zonal capacity expansion findings, interregional
transfer capacity is used as a guide for transferring power between regions to
maintain practical realizations of HVDC expansion. 62 Hence, there is a distinct
prioritization of large interregional zonal transfers (>10 GW) followed by
interregional corridors (2–10 GW) and then other potentially smaller corridors
(<2 GW).
3. Stage 1 (nodal production cost model, initial round): Set interface limits
(similar to the approach outlined in Figure 6) but using existing interface limits
from zonal capacity expansion findings. Use an initially large HVDC transfer
capacity for the HVDC links portion of the scenario (chosen as 10 GW) and run
the nodal production cost model.
4. Stage 1 (nodal production cost model, next round): Adapt the HVDC overlay
limits (from the initially chosen large capacity, 10 GW), undertake potential
relocation of termination points of the HVDC overlay as well as scaling the
relative size to established building blocks based on the use of the HVDC overlay
and rerun the nodal production cost model.
5. Stage 2 and Stage 3: Using the findings from the previous steps, undertake
HVAC transmission capacity expansion to support the HVDC overlay as defined
in the previously discussed workflow in Figure 6 to ensure secure operations and
contingency performance for HVAC expansions and large HVDC expansions. As
part of this step, there is the potential need to further refine the HVDC overlay
topology.

A.7 Snapshot Selection Methodology
When using a power flow model—either as an aid to speed up feedback to model
changes or to perform contingency screening—it is necessary to select a set of
snapshots to perform the calculation. There are many methods to do this; next is a
description of two that were used as part of this study.
Systemwide and region-specific statistics
Table A-3 shows the snapshots characteristics used in the DC power flow transmission
planning phase. Using these snapshots resulted in several hundred snapshots per
scenario being used to inform the flow impacts of adding different lines.

Building many short-distance HVDC links integrated into a multiterminal HVDC overlay is unlikely to be
practical considering the 2035 model year for the zonal-to-nodal translated scenarios—hence, the
prioritization approach taken for the zonal-to-nodal translation with respect to HVDC deployment.
62

National Transmission Planning Study

88

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Table A-3. Selection of Operating Conditions (“snapshots”) From Nodal Production Cost Model for Use in
DC Power Flow Transmission Expansion Planning Step
CONUS
Load
•
High load periods
Renewable energy production
•
Peak/high wind + solar
•
Peak/high wind
•
Peak/high solar
Instantaneous renewable energy
•
High VRE share in high load period
•
High VRE share in low load period

Region-Specific
Interregional exchanges
•
Average period for each regional interface
•
High inter-tie use (P90)
Regional balancing
•
High import (P90)
•
High export (P90)
•
High load
Transit flows
•
High transit flows across “balanced” regions

Note: Region-specific refers to transmission planning subregions.

Flow-based statistics
The following methodology is used to select the snapshots for the Stage 3 contingency
analysis in the ReEDS earlier scenario results. The premise of this approach is the
snapshots are used for a specific task: to investigate the impact of topology changes on
how flow is distributed through the system. In Stage 2, this is because of a change of
topology—for example, new transmission lines but fixed injections. In Stage 3, this is
because of a line outage and no change of injections. This observation suggests a
sample of snapshots whose flow distribution matches the flow distribution over the full
year would capture the behavior of interest.
The flow metric used here is the average flow; however, this can be adapted to U75, or
other metrics.
Consider a set of branches 𝑏𝑏 ∈ ℬ, with ratings 𝑟𝑟𝑏𝑏 , and a set of time instances 𝑡𝑡 ∈ 𝒯𝒯,
which for a year is [1,8760]. The goal is to select a subset of time 𝒦𝒦 ⊂ 𝒯𝒯. Define 𝑓𝑓(𝑏𝑏, 𝑡𝑡)
as a function that returns the flow on branch 𝑏𝑏 at time 𝑡𝑡, 𝑝𝑝(𝑏𝑏, 𝒯𝒯) as a function that
returns all time instances 𝑡𝑡 ∈ 𝒯𝒯, where the flow on 𝑏𝑏 is positive and similarly 𝑛𝑛(𝑏𝑏, 𝒯𝒯) as a
function returning all time instances where flow on 𝑏𝑏 is negative. Crucially, note all flows
are known a priori because a solved production cost model is available.
The average positive and negative flows on branch 𝑏𝑏 are calculated as
1
  𝑓𝑓(𝑏𝑏, 𝑡𝑡) ,
𝑟𝑟𝑏𝑏 |𝒯𝒯|

𝑏𝑏 − =

1
  𝑓𝑓(𝑏𝑏, 𝑡𝑡).
𝑟𝑟𝑏𝑏 |𝒯𝒯|

1
  𝑢𝑢𝑡𝑡 ⋅ 𝑓𝑓(𝑏𝑏, 𝑡𝑡) ,
𝑟𝑟𝑏𝑏 𝐾𝐾

𝑏𝑏−⋆ =

1
  𝑢𝑢𝑡𝑡 ⋅ 𝑓𝑓(𝑏𝑏, 𝑡𝑡).
𝑟𝑟𝑏𝑏 𝐾𝐾

𝑏𝑏 + =

𝑡𝑡∈𝑝𝑝(𝑏𝑏,𝒯𝒯)

𝑡𝑡∈𝑛𝑛(𝑏𝑏,𝒯𝒯)

These values are combined for all branches into a vector 𝑏𝑏 ⃗. Similarly, the estimated
branch flows based on the snapshots are defined as
𝑏𝑏+⋆ =

𝑡𝑡∈𝑝𝑝(𝑏𝑏,𝒯𝒯)

National Transmission Planning Study

𝑡𝑡∈𝑛𝑛(𝑏𝑏,𝒯𝒯)

89

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Here, 𝐾𝐾 = |𝒦𝒦| is the number of desired snapshots and 𝑢𝑢𝑡𝑡 is a binary variable (8,760
variables in total). All these variables are combined into a single vector, 𝑏𝑏 ⃗⋆ . So only 𝐾𝐾
snapshots are selected, the following constraint is enforced:

Finally, the objective function is

  𝑢𝑢𝑡𝑡 = 𝐾𝐾.
𝑡𝑡∈𝒯𝒯

Minimize𝑏𝑏 ⃗⋆ ,𝑢𝑢 ⃗  𝑏𝑏 ⃗⋆ − 𝑏𝑏 ⃗ 1

minimizing the 𝐿𝐿1 norm of the difference between the vector of average flows over all
time instances and the one calculated for just the selected 𝑡𝑡 ∈ 𝒦𝒦. The selected
snapshots are those where 𝑢𝑢𝑡𝑡 = 1 at the end of the optimization.

A.8 Alternative Approach for Contingency Analysis for Western
Interconnection Focused Analysis

The methodology for contingency analysis for the earlier scenarios for the Western
Interconnection (results presented in Section 4) is modified from that explained in
Section 2.3. The method employed incorporates single contingencies into the
production cost model, turning the formulation into a security-constrained one. First, a
DC-power-flow-based contingency analysis is performed as a screen on selected single
contingencies, as described previously. The violations are pared down to the worst
violations that must be addressed. For example, if a particular line overloads in several
snapshots by a significant amount but is otherwise not very highly used, the question of
where the right place to add transmission quickly becomes rather complex. The result is
a limited set of security constraints comprising a handful of single line contingencies
with a small number of monitored affected elements per contingency. These are added
to the production cost model and a new year simulation is performed A benefit of the
security-constrained approach is the results contain congestion costs associated with
the contingency constraints, which makes it possible to sort and rank the constraints
based on their impact to the problem objective. The costliest security constraints are the
most valuable to upgrade for the system. Where the congestion cost is low, the security
constraint can be maintained, and no further action taken. This methodology is
demonstrated on the Western Interconnection cases, because the GridView tool—used
to solve these models—can incorporate such security constraints. Note as the case
sizes increase, so does the computational burden of contingency constraints.
Depending on the number of security constraints considered and the model size, there
may be a point where the computational costs outweigh the benefits of this approach.

A.9 Impact of Branch Monitoring on Transmission Expansion
Workflow in Earlier ReEDS Scenarios
For the simulations conducted on the Western Interconnection footprint only (Limited
and AC scenarios), the limits of all branches with voltage rating 230 kV and above were
enforced, or approximately 4,000 branches. In the Multiterminal (MT)-HVDC scenario,
National Transmission Planning Study

90

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

the limits on all branches with voltage rating 345 kV and above are enforced, as well as
select 230-kV branches observed to violate their limit, for a total of ~6,000 branches. All
flows on branches 230 kV and above are recorded, however, and as shown in Figure A3 the resulting violations are not very significant (~1%).

Figure A-3. Branch loading in the MT-HVDC scenario
Chart shows branches rated ≥200 kV and ≥250 megavolts-ampere (MVA) that violate their nominal rating. Despite not enforcing all
branches as in the Western Interconnection only cases, the number of violations remains quite small (at only 1%).

The number of enforced lines in the earlier scenarios is a factor of 2–4 greater than in
the final scenarios. This results in greater sensitivity of generator dispatch to the
changing topology, i.e., transmission expansion. A consequence of the increased
sensitivity is the DC power flow step in Stage 2 (cf. Section 2.3) of the transmission
expansion becomes more challenging to interpret because an underlying assumption of
iteration with the DC power flow is an unchanged generation dispatch. Therefore, the
DC power flow is not relied on in Stage 2 in the earlier ReEDS scenarios. On the other
hand, in Stage 3, where select single contingencies are considered, the DC power flow
is still used extensively as a valuable contingency screening tool. The methodology for
selecting representative snapshots is described under flow-based statistics in this
appendix.

A.10 Design Approach for a Nationwide MT-HVDC System in Earlier
ReEDS Scenario (demonstrated in Section 4)
Modeling large MT-HVDC systems, embedded in AC interconnects, in production cost
models is an emerging area (Nguyen et al. 2024). The flows within each MT-HVDC
system are determined by shift factors, very similar to the power transfer distribution
factors (PTDFs) in the linearized AC system. Converter models are added as control
variables that link AC and DC buses.
The objective of the MT-HVDC scenario is to explore the implications of a large-scale
MT-HVDC system covering a large portion of the country. A design decision on the
earlier ReEDS scenarios is to use a single design that would facilitate combining
segments. All new HVDC lines modeled in this scenario are ±525-kV voltage source
National Transmission Planning Study

91

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

converter (VSC) bipole design with conductor parameters adapted from the CIGRE
Working Group (WG) B4.72 DC Benchmark Models (CIGRE WG B4.72 2022, 804). The
rating of a single bipole does not exceed 4.2 GW.
Converter stations are sited based on the underlying AC system. They are embedded in
the AC system at comparatively strong/well-meshed locations to enable delivery to/from
the DC system as well as provide a viable alternative path during contingency events.
Converter locations can be grouped into three categories: VRE hubs, load centers, and
connection points to AC transmission hubs.
HVDC transmission is generally chosen for two key reasons: high power transfer over
large distances and flow controllability. MT-HVDC systems have the benefit of fewer
converters than point-to-point systems and therefore lower capital costs; meshed MTHVDC systems have the additional advantage over radial systems of reliability under
failure because of alternative current paths. Reduced converter count in MT-HVDC
systems comes at the cost of increasing mismatch between line and converter rating,
depending on whether each converter on the line operates as a source or a sink.
Reliability in meshed MT-HVDC systems comes at the cost of some loss in
controllability because the flow distribution between parallel paths will be the function of
shift factors, similar to the AC system.
The design choice in the MT-HVDC scenario presented in Section 4 of this chapter is a
compromise between the costs and benefits of the three frameworks described
previously. First, several MT-HVDC systems that are largely radial are created. At their
intersections, they do not share a DC bus but rather an AC bus between a pair of
converters, 63 as shown in Figure A-4. This solution sacrifices some of the cost saving of
MT-HVDC, because of the additional converter, but gains back more controllability while
maintaining the reliability of a system with multiple paths for power to flow. The final
HVDC topology is presented in Figure A-5, which shows HVDC expansion color coded
to highlight the separate MT-HVDC systems.

63

Note a converter here actually refers to the converter pair ±525 kV, given the bipole design.

National Transmission Planning Study

92

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure A-4. Coupled MT-HVDC design concept and rationale

Figure A-5. HVDC buildout in the MT-HVDC scenario
Different colors correspond to separate MT-HVDC systems interconnected via the AC system as illustrated in Figure A-4.

National Transmission Planning Study

93

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Appendix B. Scenario Details
B.1 New Transmission Network Elements
Table B-1. Nodal Transmission Building Block Characteristics (overhead lines)
Voltage (kilovolts
[kV])

Conductor

Pos.
Seq. R
(ohms
per
kilometer
[Ω/km])

Pos. Seq. X
(Ω/km)

Rate
(MVA)

Source

230 kV

2 x Bluejay

0.0299

0.3462

703

(1)

345 kV

2 x Bluejay

0.0299

0.3579

1,055

(1)

345 kV

3 x Bluejay

0.02

0.3127

1,566

(1)

500 kV

4 x Grosbeak

0.0259

0.3145

2,187

(1)

500 kV

4 x Bluejay

0.0152

0.3105

3,027

(1)

500 kV

6 x Bluejay

0.0102

0.2186

3,464

(1)

765 kV

6 x Bluejay

0.0105
0.0099

0.2831

5,300

(1)

N/A

2,100

(2)

HVDC (525 kV)

Bobolink

Sources:
Empresa de Pesquisa Energética (EPE) (2022)
CIGRE WG B4.72 (2022)

Table B-2. Nodal Transmission Building Block Characteristics (transformation capacity)
Voltage
(kV)

X
(p.u. on Transformer MVA base)

Rate
(MVA)

345/230

0.12

2,000

500/230

0.12

2,000

500/345

0.12

2,000

765/500

0.12

2,000

In the earlier ReEDS scenario results, an analysis of transformers in the Western
Electricity Coordinating Council (WECC) starting case (2030 Anchor Dataset [ADS])
shows the relationship between transformer MVA base and per unit reactance on the
system basis could be approximately described as
𝑥𝑥[p.u.] = 0.0523 ln(MVA) − 0.2303.

This formula is used to calculate variable reactance, under the assumption the
transformer base and rating are equal.
Converter losses are estimated on a per unit basis as
𝑅𝑅[p.u.] = loss% /𝑃𝑃Rated[MW]
National Transmission Planning Study

94

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Where 0.7% is assumed for line-commutated converter (LCC) technology and 1.0% for
voltage source converter (VSC) technology. Though the production cost models are
conducted without losses, these values are important for the downstream power flow
models in Chapter 4, where a full alternating current (AC) power flow that includes
losses is conducted.

National Transmission Planning Study

95

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

B.2 Region Definitions

Figure B-1. Subtransmission planning regions (derived and aggregated from Regional Energy Deployment System [ReEDS] regions)

National Transmission Planning Study

96

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure B-2. Subregion definitions used in Section 4 (earlier ReEDS scenario results)
The definition of regions is based on the underlying 134 ReEDS regions (excluding Electric Reliability Council of Texas [ERCOT] regions).

National Transmission Planning Study

97

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

B.3 Detailed Motivation for Transmission Expansion for Earlier
ReEDS Scenarios
The following sections describe the key transmission additions and their rationale on a
regional basis. For the Western Interconnection footprint, comparisons between the
scenarios are given whereas for the Eastern Interconnection regions, the description is
for the Multiterminal (MT) High-Voltage Direct Current (HVDC) scenario only.
California (CALS and CALN)
All scenarios have some intraregional transmission expansion, particularly in southern
California to help the large amount of solar power get to load. Northern California has a
few 500-kilovolt (kV) backbone changes associated with offshore wind integration that
are based on plans in California Independent System Operator’s (CAISO’s) 20-year
outlook. 64
In the interregional scenario, the connection to the Desert Southwest (DSW) and Basin
region is strengthened both via Nevada as well as along the southern border. This is
mirrored in the MT-HVDC scenario with an MT-HVDC connection to the Lugo
substation. In the interregional scenario, a 500-kV backbone on the eastern part of the
state is added to increase the transfer capacity between northern and southern
California. In all cases, the transmission is primarily motivated by bringing external
resources into California, primarily eastern wind. However, the transmission capacity
also serves to export California solar energy during the daily peaks.
Northwest (NWUS-W and NWUS-E)
In all scenarios, some form of a 500-kV collector system is added to collect wind in
Montana and attach it to the Colstrip 500-kV radial feed. In the two AC cases, a new line
from Garrison to Midpoint is used to provide an alternative export to WECC Path 8 and
a connection to the Gateway projects. In the MT-HVDC case, WECC Path 8 is
effectively reinforced with an MT-HVDC line connecting North Dakota and the Ashe
substation in Washington, with a terminal in Colstrip in between.
Desert Southwest (DSW)
The desert southwest is one of the more varied regions in terms of transmission, driven
by very different resource buildouts between the scenarios. In the limited AC scenario,
the region has substantially (≥20 GW) more solar capacity installed than the other two
cases. The 500-kV system along the southern border is reinforced and expanded to
collect that solar energy and bring it to load. In the interregional scenario, substantially
more wind (~11 GW) is installed, predominantly in New Mexico, necessitating a
significant 500-kV backbone expansion. The energy is provided in three main paths:
north via Four Corners, west via Coronado, and south toward Phoenix. The MT-HVDC
scenario has a lower solar build, like the interregional AC scenario, and a wind build
somewhere between the Limited and interregional scenarios. As a result, the AC
expansion is not as extensive as the interregional scenario but taken together with the
64

https://stakeholdercenter.caiso.com/RecurringStakeholderProcesses/20-Year-transmission-outlook

National Transmission Planning Study

98

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

DC scenario forms a similar three-corridor design to get both New Mexico wind as well
as wind from across the seam to the Eastern Interconnection toward the western load
pockets.
A commonality between all scenarios is the reinforcement of the 500-kV corridor across
northern Arizona into Nevada. This is driven by the ability of the transmission to carry
both solar from southern Utah/northern Arizona that is present in all scenarios as well
as the wind out of New Mexico via Four Corners when present.
Rocky Mountain (ROCK)
The 345-kV system around Denver is expanded to bring in predominantly wind
resources. To the south, the Colorado Power Pathway 65 is part of the augmented
starting case and is expanded on in the scenarios. To the north, connections to the
Cheyenne and Laramie River region in Wyoming are added. Two paths are generally
used—one to the Public Service Company of Colorado and one to the Western Area
Power Administration systems.
The Gateway projects are also added in all scenarios as part of the augmented starting
case. They are expanded in all scenarios to collect more of the Wyoming wind. In the
limited AC case, with less wind, this is done with mainly 230-kV reinforcements. In the
interregional AC and MT-HVDC cases, a northern 500-kV section is added between the
Windstar and Anticline substations.
The TransWest express 66 DC and AC projects are incorporated into the augmented
starting case and bring wind from Wyoming toward the Desert Southwest and
California.
The Rocky Mountain region is loosely connected to the rest of the Western
Interconnection compared to the other regions. One of the key differences between the
limited AC scenario and the others is the connection between the Rockies and other
regions. In the interregional AC scenario, there is a connection to New Mexico via the
500-kV backbone built to collect wind. In the MT-HVDC scenario, an HVDC backbone
connects the Rocky Mountain region to the Basin via Wyoming, to the Desert Southwest
via New Mexico, and to Southwest Power Pool (SPP) to the east.
Basin (BASN)
In the modeled scenarios, the Basin region plays a key role in providing multiple paths
for resources to reach load. The projects involving the Basin region largely begin/end in
other regions: Montana, Rocky Mountain, Desert Southwest, Pacific Northwest, or
California and are therefore not repeated here. The MT-HVDC scenario builds on this
connector role for the Basin with two east-west corridors: one toward the Pacific
Northwest and one toward the Bay Area in California.

65
66

https://www.coloradospowerpathway.com/
https://www.transwestexpress.net/

National Transmission Planning Study

99

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

SPP
The transmission expansion in SPP is focused on the east and west movement of wind
but some solar in the southern areas as well. To that end, multiple HVDC corridors
begin in the SPP footprint and move east or connect across the seam to the Western
Interconnection. In addition, a north-south corridor connects to Midcontinent
Independent System Operator (MISO) and allows resources to shift between the
various east-west corridors. Finally, the wind in the southern part of SPP is intended to
complement solar in both Florida and California via an MT-HVDC corridor stretching
from coast to coast along the southern part of the country.
In addition to the MT-HVDC, a complementary 345-kV collector backbone is added
along the north-south axis on the SPP footprint. The intention is to both collect the VRE
resources and provide an alternative path under contingency events.
A 500-kV loop interconnects with MISO in the southern end of the footprint as an
alternative path east for wind resources in SPP and to provide better access to load and
the MT-HVDC system for solar resources in MISO.
MISO
An MT-HVDC system runs north-south along MISO’s eastern portion to provide
exchange between solar- and wind-rich regions as well as a path to the Chicago, Illinois
load center. The east-west MT-HVDC corridors originating in SPP include multiple
terminals in MISO to serve load and collect further resources toward the east. The eastwest corridors terminate at three types of destinations: load centers in PJM and the
southeast, solar hubs in PJM and the southeast, or strong 765-kV substations in PJM’s
footprint that can carry the power farther east to the coast.
In terms of AC expansion, Tranch 1 67 additions are added to the MISO footprint along
with further reinforcements of the 345-kV system.
PJM
The MT-HVDC connections to PJM are largely described in the SPP and MISO
sections. It is noted here Chicago serves as an MT-HVDC hub for several east-west
lines as well as the north-south MISO system. Finally, the north-south SPP MT-HVDC
system loops east and terminates in PJM between Toledo, Ohio and Cleveland, Ohio.
Some of the 765-kV segments of the PJM system are upgraded to enable eastward
transfer of more power coming from the MT-HVDC systems.
Along the Eastern Seaboard, there is a new DC link across the Chesapeake Bay and a
230-kV collector system for solar and offshore wind between Delaware and Maryland.

https://cdn.misoenergy.org/MTEP21%20AddendumLRTP%20Tranche%201%20Report%20with%20Executive%20Summary625790.pdf
67

National Transmission Planning Study

100

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Southeast
Beyond the MT-HVDC systems terminating in the southeast, as described in the SPP
and MISO sections, 500-kV reinforcements are built to connect predominantly solar
resources to load or the MT-HVDC overlay. Examples are 500-kV connections between
Panama City, Mobile, and Montgomery, Alabama and lines from Charlotte, North
Carolina down to Augusta, South Carolina.
FRCC
An MT-HVDC link connects FRCC to the southeast and farther west, intended to import
wind from the middle of the country during the night and export the Florida sun in the
middle of the day. In addition, a 500-kV backbone is extended from the central Florida
region north to further collect solar resources in the northern part of the state and get
them toward load centers on the coasts and to the south.
NYISO
A new HVDC link connects Long Island, New York to Connecticut. Several 345-kV
upgrades are also performed on Long Island to accommodate offshore wind projects.
The Northern NY Priority, Champlain-Hudson Power Express, and Clean Path NY are
also added by default to augment the starting case.
ISONE
Here, 345-kV collector systems are added in Maine, New Hampshire, and Vermont to
collect new wind and solar resources and connect them to the existing 345-kV
backbone.

National Transmission Planning Study

101

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

B.4 Further Detailed Results for Final Scenarios

Zonal (2020)

Limited (2035)

AC (2035)

MT- HVDC (2035)

Figure B-3. Interregional transfer capacity from zonal ReEDS scenarios
Aggregated to transmission planning regions (some larger regions are further subdivided to aid in planning).

National Transmission Planning Study

102

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Detailed installed capacity (by scenario)

Figure B-4. Summary of installed capacity by transmission planning region, interconnection and
contiguous U.S. (CONUS)-wide

National Transmission Planning Study

103

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Nodal scenarios transmission expansion summary

Figure B-5. Summary of nodal transmission expansion portfolios
Shown for AC, Limited, and MT-HVDC scenarios in terms of circuit-miles (left), thermal power capacity (middle), and terawatt-miles
[TW-miles] (right).

Detailed nodal scenarios transmission expansion motivation
Table B-3. Rationale for Nodal Transmission Portfolios (Limited scenario)
Interconnect

Transmission
Planning Region1

Eastern
Interconnection

FRCC

Summary
-

Eastern
Interconnection

ISONE

Eastern
Interconnection

MISO

-

National Transmission Planning Study

Significant solar photovoltaics (PV) expansion in the centraleastern side of Florida with large amounts of existing natural
gas capacity remaining online
Localized integration of new solar PV capacity does not
require interregional transmission expansion with the
Southeast
Exiting 500-kV and 230-kV networks suffice for this scenario
Predominantly solar PV and offshore wind integration to the
southern parts of New England
No new 345-kV expansions necessary with New York
Independent System Operator (NYISO)
Inclusion of MISO Long-Range Transmission Planning
(LRTP) Tranche 1 (several new 345-kV lines) across
MISO-N and MISO-C (prescribed expansion)
345-kV reinforcements in southern part of MISO-C and 500kV reinforcements in MISO-S to accommodate integration of
new wind and solar PV capacity
Additional 345-kV reinforcements in weak parts of MISO-N
to strengthen existing network to integrate predominantly
new wind capacity and move power east-west toward load
centers in MISO-N and PJM-W

104

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Interconnect

Eastern
Interconnection

Transmission
Planning Region1

NYISO

Summary
-

Reinforcement of selected parts of the existing 765-kV
network in MISO-C to integrate with PJM-E and PJM-W

-

Addition of prescribed transmission (Northern NY Priority,
Champlain-Hudson Power Express, and Clean Path NY)
Strengthening of existing 500-kV links in NYISO-NYC with
PJM
345-kV expansion to move power across to Long Island
Selected parts of NYISO-NYC 230-kV localized
strengthening of existing circuits to enable solar PV
integration
345-kV strengthening NYISO-UP (upstate) from Buffalo
toward PJM (north-south) and new 345-kV along the edges
of Lake Erie toward Cleveland

-

Eastern
Interconnection

PJM

-

-

-

Eastern
Interconnection

Southeast2

-

-

Eastern
Interconnection

SPP

-

-

National Transmission Planning Study

Development of a 500-kV overlay (double-circuit) in
Maryland/Delaware area to support integration of large
volumes of offshore wind and solar PV
Strengthening of existing single-circuit 500-kV paths (northsouth) in New Jersey and Pennsylvania and parts of Virginia
East-west Pennsylvania 500-kV strengthening of existing
single-circuits increasing transfer capacity between 345-kV,
500-kV, and 765-kV networks
Strengthening of many of the existing single-circuit 765-kV
network in the PJM-East footprint (Ohio, West Virginia,
Virginia) to increase transfer capacity with MISO-Central and
PJM-West footprints
Selected strengthening of 345-kV networks in single-circuit
to double-circuit at the confluence of the 345-kV, 500-kV,
and 765-kV networks in PJM-East as well as interregionally
with MISO-Central
Imports into the PJM-West footprint are enabled by some
single-circuit to double-circuit strengthening of 345-kV
networks within PJM-West and links to MISO-Central
Large amounts of meshed single-circuit to double-circuit
500-kV network strengthening in Tennessee, Mississippi,
and Alabama (less strengthening in Georgia and the
Carolinas)
Several new double-circuit 500-kV between Alabama and
Georgia
Relatively strong 230-kV networks in the Carolinas and
Georgia enable integration of expected solar PV, but some
230-kV single-circuit to double-circuit strengthening is
needed in southern parts of Alabama and South Carolina
Reinforcement of 345-kV network north-south in SPP-North
to accommodate integration of new wind capacity
Some localized 230-kV strengthening to collect relatively
large amounts of wind capacity
Expansion of 345-kV network in SPP-South to integrate
large amounts of wind and solar PV capacity and move
power west-east and further into the central parts of SPP
toward seams with MISO-Central
Pocket in SPP-South with large amounts of wind and solar
PV (Texas panhandle and eastern edge of New Mexico)

105

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Interconnect

Transmission
Planning Region1

Summary
require substantial 230-kV strengthening and 345-kV exports
north toward Kansas and west-east to Oklahoma

Western
Interconnection

CAISO

-

Western
Interconnection

NorthernGrid

-

-

Western
Interconnection

WestConnect

-

-

ERCOT

ERCOT

-

-

CAISO-North expansion of 500-kV to integrated offshore
wind and solar PV combined with strengthening of existing
single-circuit 500-kV routes
Selected 230-kV expansions in CAISO-North to integrate
solar PV and enable power to move south toward load
centers
In addition to prescribed Boardman-Hemingway, Greenlink
Nevada, and TransWest Express, additional 500-kV doublecircuit between Wyoming and Montana is added to move
large amounts of wind capacity from Wyoming
Strengthening of 500-kV path on the NorthernGrid West
edge (single-circuits to double-circuits) in Oregon for parts of
offshore wind integration and as an additional path for power
from wind in Wyoming and Idaho
500-kV expansion from Wyoming north-south toward
Colorado, including extensive 230-kV strengthening to
collect large amounts of wind capacity
Further 345-kV and 230-kV expansion in CO (in addition to
Colorado Power Pathway) to enable further paths for large
amounts of wind capacity farther south toward New Mexico
and Arizona
345-kV strengthening in WestConnect South (single-circuits
to double-circuits) moving wind and solar PV capacity eastwest toward load centers in Arizona and California
Additional 500-kV path created between New Mexico and
Arizona also to move large amounts of wind and solar
capacity
Several new double-circuit 345-kV expansions from West
Texas toward Dallas-Fort Worth (west-east) and southeast
toward San Antonio and Austin
Further strengthening of existing 345-kV single-circuit routes
similarly moving large amounts of wind capacity toward load
centers
Northern parts of ERCOT also further strengthened with new
double-circuit 345-kV expansions also moving large
amounts of wind capacity from northern parts of ERCOT and
solar PV from eastern parts of Texas toward load centers in
Dallas-Fort Worth
Several new double-circuit 345-kV north-south expansions
between Dallas and Houston and links to San Antonio and
Austin created to further move west Texas wind capacity and
solar in south Texas to load centers

1

Using transmission planning regions mapped from 134 planning regions in ReEDS (capacity expansion tool).

2

SERTP/SCRTP

National Transmission Planning Study

106

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Table B-4. Rationale for Nodal Transmission Portfolios (AC scenario)
Interconnect

Transmission
Planning Region1

Eastern
Interconnection

FRCC

Eastern
Interconnection

ISONE

Summary
-

Large amounts of solar PV capacity are supported by
relatively strong existing 230-kV networks, but additional
strengthening in central and southern parts of the Florida
Reliability Coordinating Council (FRCC) footprint are needed

-

New double-circuit 500-kV interregional links created to the
Southeast region (Florida to Georgia) to enable large
interregional power flows between FRCC and the Southeast

-

Strengthening of existing single-circuit 345-kV to doublecircuit 345-kV between Independent System Operator of
New England (ISONE) and NYISO (upstate) –
Massachusetts – New York, Connecticut – New York
Additional 345-kV strengthening between Maine and
New Brunswick
Large amount of 345-kV expansion (north-south) along Lake
Michigan to move substantial wind capacity toward load
centers

Eastern
Interconnection

MISO

National Transmission Planning Study

-

-

Large amounts of further 345-kV strengthening in MISONorth on seams with SPP-North

-

Two new 500-kV double-circuit overlays (west-east) in
MISO-North up to SPP-North seams to move large wind and
some solar PV capacity across Minnesota and Wisconsin as
well as Iowa and Nebraska to load centers in Illinois

-

Two new 500-kV double-circuit overlays in MISO-Central
with additional double-circuit (north-south) to linking MISOCentral and SPP-South moving large amounts of wind
capacity (predominantly for west-east transfers) but with
supplementary new 500-kV overlay (north-south) improving
contingency performance from Kansas to Missouri and
Illinois

-

MISO-Central requires substantial strengthening of the
existing 345-kV networks (in addition to prescribed LRTP
Tranche 1 projects) to integrate substantial wind and
solar PV capacity while creating critical enabling links
between SPP and PJM-East footprint (mostly single-circuit
to double-circuit expansion)

-

Strengthening of the existing 765-kV networks in Indiana
(MISO-Central) as a backbone for moving wind and solar PV
into PJM-West as well as into PJM-East footprints; no new
765-kV rights-of-way required

-

Large amounts of wind capacity are integrated and moved
west-east via new 500-kV voltage overlays in MISO-South to
interconnect with SPP-South and the Southeast region as
well as strengthening of existing 500-kV ties with MISOSouth and the Southeast

107

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Interconnect

Eastern
Interconnection

Transmission
Planning Region1

NYISO

Summary
-

Substantial amount of 230-kV strengthening in MISO-North
(Minnesota and North Dakota); all single-circuit to doublecircuit expansions

-

New double-circuit 345-kV overlay (from 230-kV) in NYISO
combined with strengthening of existing 345-kV in NYISO
(upstate) Buffalo area with PJM-West
Further strengthening of existing 345-kV interregional link
between NYISO (updates) and PJM-West

Eastern
Interconnection

PJM

-

-

-

-

Eastern
Interconnection

Southeast2

National Transmission Planning Study

-

Development of a 500-kV overlay (double-circuit) in
Maryland/Delaware area to support integration of large
volumes of offshore wind and solar PV
Large amounts 500-kV single-circuit to double-circuit
strengthening along eastern part of PJM-East (Virginia,
Pennsylvania, Maryland) enabling further offshore wind
integration and moving power—predominantly solar
capacity—from the Southeast into PJM-West
Imports into the PJM-West footprint are enabled by some
single-circuit to double-circuit strengthening of 345-kV
networks within PJM-West and links to MISO-Central
Interregional links strengthened via 345-kV and 500-kV
strengthening with NYISO
Central parts of Pennsylvania 345-kV and 500-kV
confluence strengthened further (single-circuit to doublecircuit)
Many of the existing 765-kV circuits in PJM-West
strengthened to enable bulk movement of power
interregionally between MISO-Central and the Southeast as
well as integration and key links with strengthened 345-kV
and 500-kV networks
500-kV voltage overlay and meshing of existing 345-kV
networks in Kentucky to increase transfer capacity with the
Southeast region
PJM-West footprint includes strengthening of existing 345kV circuits (double-circuits) to enable imports from MISOCentral and MISO-North in addition to 500-kV voltage
overlay from MISO-North
Several interregional ties with PJM-East region are further
strengthened in addition to new 500-kV double-circuit
expansions (North Carolina, Virginia, Kentucky, Tennessee)

-

Strengthening of existing 500-kV ties with MISO-South to
move solar and wind capacity bidirectionally between
Southeast and MISO-Central/MISO-South (Tennessee,
Arkansas, Mississippi, Louisiana, Alabama)

-

Construction of an additional 500-kV circuit across existing
500-kV ties with FRCC

-

Large number of 230-kV intraregional strengthening in the
Carolinas considering the large amount of solar PV and wind
capacity expanded

108

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Interconnect

Transmission
Planning Region1

Eastern
Interconnection

SPP

Western
Interconnection

Western
Interconnection

CAISO

NorthernGrid

National Transmission Planning Study

Summary
-

Large 500-kV voltage overlays in SPP-South (from 345-kV)
across five paths into the Texas panhandle, across
Oklahoma, up into Kansas and west-east toward Missouri
enabling interregional expansion and movement of large
amounts of wind capacity to Southeast and MISO-Central.

-

Similar 230-kV expansions in SPP-South to the Limited
scenario in Texas panhandle and parts of New Mexico to
export wind capacity north and east toward Oklahoma,
Kansas, and Missouri

-

500-kV overlay (from 345-kV) in SPP-North to move large
wind and solar PV capacity interregionally to MISO-North
(west-east)

-

Extensive 345-kV strengthening in SPP-North (multiple
paths) moving power southeast to link with MISO-North 345kV networks linking to the 500-kV overlay in MISO-North

-

New 500-kV double-circuit overlays (from 345-kV)
increasing interregional transfer capabilities between SPPNorth and MISO-North

-

New double-circuit 500-kV paths (x2) in CAISO-North to
integrate offshore wind backbone into existing 500-kV northsouth paths

-

Expansion of single-circuit 500-kV circuits to double-circuit
500-kV in main interregional transfer corridor between
California and Oregon

-

New 500-kV double-circuit expansions between CAISONorth and CAISO-South

-

Extensive strengthening of existing 500-kV paths from
Nevada (x4) into California (CAISO-South) enabling imports
of large amounts of wind capacity and solar PV capacity
from WestConnect and NorthernGrid-South

-

500-kV overlay of existing 230-kV networks in CAISO-South
with WestConnect-South moving large amounts of solar PV
and wind capacity from Arizona and New Mexico

-

New 500-kV to expand the transfer capacity between the
West and South

-

Further strengthening of 500-kV expansions in Nevada (in
addition to TransWest Express and Greenlink Nevada) to
enable north-south transfer from NorthernGrid-East and
NorthernGrid-South

-

In addition to Boardman-Hemingway, further additional new
double-circuit 500-kV paths are created (north-south)
moving capacity through Idaho and toward Washington

-

Strengthening existing east-west 500-kV transfer capacities
between Idaho and Oregon (single-circuit to double-circuit)

109

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Interconnect

Transmission
Planning Region1

Summary
in addition to north-south 500-kV strengthening between
Oregon and Washington to enable offshore wind integration
combined with land-based wind expansion predominantly in
Idaho (additional transfer paths)

Western
Interconnection

ERCOT

WestConnect

ERCOT

-

New 500-kV overlays from 230-kV between NorthernGridEast and WestConnect (Montana – Wyoming and
Idaho – Wyoming)

-

In addition to Colorado Power Pathway projects (345-kV),
further 345-kV additions (east-west) increasing transfer
capabilities between Colorado and Utah

-

345-kV strengthening between WestConnect-North and
WestConnect-South (four-corners) as additional paths for
predominantly Wyoming and Colorado wind capacity

-

Extensive 500-kV strengthening (single-circuit to doublecircuit) in WestConnect-South predominantly enabling
moving solar PV and wind capacity from New Mexico and
Arizona to California

-

Several 500-kV overlays of existing 345-kV networks in
WestConnect-South (New Mexico and Arizona) playing a
similar role as 500-kV strengthening (moving wind and solar
PV capacity east-west to load centers in Arizona and
California)

-

Unchanged from Limited

1

Using transmission planning regions mapped from 134 planning regions in ReEDS (capacity expansion tool).

2

SERTP/SCRTP.

Table B-5. Rationale for Nodal Transmission Portfolios (MT-HVDC scenario)
Interconnect

Transmission
Planning Region1

Eastern
Interconnection

FRCC

Summary
-

-

2- x 4-GW bipoles: one FRCC and Southeast and one
between FRCC-MISO (South) used for bidirectional power
transfers of solar PV (and some wind from nonadjacent
regions including MISO and SPP)
Localized 230-kV strengthening in northern, central, and
southern parts of Florida integrating large amounts of
solar PV capacity
Single-circuit to double-circuit 500-kV expansion between
FRCC and Southeast (Florida – Georgia)

Eastern
Interconnection

ISONE

-

N/A (other than already prescribed transmission
expansion)

Eastern
Interconnection

MISO

-

2- x 4-GW bipole from northern part of MISO into PJMWest to bring wind capacity from extreme northern
(Montana) and central parts (Minnesota) of MISO-North to
load centers in PJM-West

National Transmission Planning Study

110

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Interconnect

Transmission
Planning Region1

Summary
-

Northern parts of MISO-North require some strengthening
of existing 230-kV and 345-kV networks to collect wind
capacity for HVDC transfers

-

Additional 2- x 4-GW bipoles (multiterminal) from central
parts of MISO-North into PJM-West and MISO-Central for
moving power across MISO and toward PJM-West
Combination of some new double-circuit 345-kV west-east
and strengthening of existing paths to support further
movement of wind capacity in MISO-North toward PJMWest and MISO-Central
MISO-Central requires additional single-circuit to doublecircuit strengthening (Missouri, Indiana, Illinois, Michigan)
Only selected parts of the existing 765-kV networks
between MISO and PJM require single-circuit to doublecircuit strengthening to enable transfers between HVDC
terminals in MISO and PJM
4- x 4-GW multiterminal HVDC links in MISO-South
enable linking between SPP-South, MISO-South, and the
Southeast region
1- x 4-GW HVDC links with ERCOT for exports
predominantly of wind and solar PV from ERCOT

-

-

Eastern
Interconnection

NYISO

-

Eastern
Interconnection

PJM

-

-

Eastern
Interconnection

Southeast2

-

National Transmission Planning Study

2-GW monopole between upstate NYISO and PJM
increasing exchange capacity between NYISO and PJM
Strengthening of 345-kV east-west transfer capacity
across NYISO (upstate) enabling exports from 2-GW
bipole
Additional 500-kV strengthening between PJM-West and
NYISO (link into existing 345-kV networks in NYISO)
4- x 4-GW PJM-West – PJM-East, MISO-Central – PJMEast, PJM-East – Southeast link interregionally between
MISO, Southeast, and NYISO, predominantly to move
wind capacity from MISO (and indirectly from SPP) into
PJM
Selected expansion of the existing 765-kV networks
enables exports from HVDC terminals, transfers to
embedded HVDC terminals, and improved contingency
performance in the footprint
New 500-kV overlay as in AC and Limited to integrated
offshore wind capacity
Several interregional 500-kV strengthening needs along
Eastern Seaboard and between PJM-East and Southeast
4-GW bipole linking PJM-East and Southeast to support
the combination of large amounts of wind and solar from
the Southeast to PJM
4-GW bipole moving bidirectional power between MISOCentral and the Southeast leveraging wind in MISO and
solar PV in the Southeast
4-GW bipole between the Southeast and SPP-South
moving large amounts of wind into the Southeast
Several supporting 500-kV strengthening in Tennessee,
Georgia, Alabama, and Mississippi enabling embedded
HVDC imports into the Southeast and exports to MISOCentral and PJM-East

111

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Interconnect

Transmission
Planning Region1

Eastern
Interconnection

SPP

Summary
-

-

-

Western
Interconnection

CAISO

-

Western
Interconnection

NorthernGrid

-

Western
Interconnection

WestConnect

-

-

-

National Transmission Planning Study

2- x 4-GW bipoles between SPP and PJM-West moving
large amounts of wind into strong areas of the existing
765-kV and 500-kV networks
2- x 4-GW multiterminal bipoles moving power north to
south within SPP and enabling further transfers from SPP
to MISO-Central and the Southeast
5- x 4-GW and 1- x 2-GW seam-crossing bipoles across
several locations in SPP to move power between the
Eastern Interconnection and Western Interconnection—
four of these with WestConnect and one with
NorthernGrid
WestConnect-South and SPP-South HVDC comprises the
starting point of meshed HVDC expansions between the
Eastern Interconnection and Western Interconnection
2- x 4-GW seam-crossing HVDC bipoles between SPPSouth and ERCOT mostly for bidirectional exchange of
complementary wind capacity in ERCOT and SPP-South
New 500-kV overlay west-east in SPP-South enabling
improved contingency performance of extensive HVDC
expansion in SPP-South and further transfer capacity for
wind and solar PV in Oklahoma and Texas panhandle to
MISO-South
Several supporting single-circuit to double-circuit 345-kV
expansions in SPP (South) to integrate large wind and
solar PV capacity as well as improve contingency
performance of HVDC expansions
2- x 4-GW HVDC bipoles between CAISO-South and
WestConnect-South shifting large amounts of solar PV
and wind from Arizona and New Mexico toward California
Similar new 500-KV expansions in CAISO-North for
integration of offshore wind as in AC and Limited
The strengthening of existing 500-kV between CAISONorth and CAISO-South enables further increased
transfers
1- x 2-GW seam-crossing link between NorthernGrid-East
and SPP-North
Single-circuit to double-circuit 500-kV strengthening in
NorthernGrid-East and NorthernGrid-West enables similar
transfers of wind capacity from Wyoming toward Oregon
and Washington
Offshore wind is integrated in Oregon in a similar manner
to that of the AC and Limited scenarios
2- x 4-GW seam-crossing capacity between ERCOT and
WestConnect-South as well as 4-GW bipole between
Arizona and New Mexico enables the combination of New
Mexico and Texas wind and solar PV capacity to be sent
toward load centers in Arizona and California
4- x 4-GW meshed HVDC seam-crossing expansion
between WestConnect-South and SPP-South enables a
large component of wind and solar capacity between the
Eastern Interconnection and Western Interconnection
2- x 4-GW seam-crossing capacity expansion between
WestConnect-North and SPP-North provides for the

112

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Interconnect

Transmission
Planning Region1

Summary

-

-

-

ERCOT

ERCOT

-

-

remainder of the dominant transfer capability between the
Eastern Interconnection and Western Interconnection
Two additional 4-GW HVDC links in WestConnect-North
move Wyoming wind toward Colorado and across the
seam-crossing HVDC links between WestConnect-North
and SPP-North
4-GW bipole as seam-crossing capacity from
WestConnect (New Mexico) to the southernmost parts of
SPP
4-GW bipole moving power within WestConnect toward
Colorado and enabling of seam-crossing between
WestConnect and SPP-South
Extensive strengthening of the existing 345-kV and 500kV networks in WestConnect-South predominantly in
New Mexico and Arizona support the multiterminal HVDC
links between WestConnect, CAISO, and ERCOT
New 500-kV overlay (from 345-kV) and new 500-kV paths
between Arizona and New Mexico further supporting
HVDC transfers and collection of wind/solar PV capacity
2- x 4-GW seam-crossing HVDC bipoles between ERCOT
and the Western Interconnection (WestConnect)
predominantly moving large amounts of wind and solar
power from ERCOT to WestConnect; this supports
exports from West Texas and hence a decreased need for
345-kV strengthening or new 345-kV paths
2- x 4-GW seam-crossing HVDC bipole between ERCOT
and MISO-South getting wind from ERCOT into the
Eastern Interconnection
2- x 4-GW seam-crossing HVDC bipole between ERCOT
and SPP-South similarly getting wind and solar PV
capacity from ERCOT into the Eastern Interconnection
Existing 345-kV strengthening (single-circuit to doublecircuit) is concentrated between Dallas-Fort Worth and
Houston (north-south) with selected new paths needed
and enabled by new double-circuit 345-kV expansions

1

Using transmission planning regions mapped from 134 planning regions in ReEDS (capacity expansion tool).

2

SERTP/SCRTP.

National Transmission Planning Study

113

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

B.5 Further Detailed Results for Earlier Scenarios
Curtailment results in the Western Interconnection
Earlier scenarios (Section 4) used curtailment as a key stopping criteria metric during
evaluation of the nodal translations. Figure B-6 compares aggregate curtailment
numbers from the three scenarios. Curtailment peaks in the second quarter of the year,
corresponding to the hydro runoff in the west, which is also a period of relatively low
demand.

Figure B-6. Curtailment comparison between scenarios in the Western Interconnection
Each panel represents a different level of spatial or temporal aggregation: (a) total curtailment, (b) curtailment by quarter, (c)
curtailment by region.

National Transmission Planning Study

114

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure B-7 shows the average weekday curtailment along with standard deviation for
the three scenarios. Solar curtailment exhibits a high standard deviation over time
because of seasonality or other weather events whereas wind is more consistent on
average. As the share of wind increases in the interregional scenarios, it begins to
exhibit more of the midday curtailment peak of solar.

Curtailment [GWh]

25
20
15
10
5

Curtailment [GWh]

0
0

5

10

15

20

10

15

20

30
20
10
0
0

5

Hour of Day

Figure B-7. Relationship between solar and wind curtailment in the Western Interconnection
Average weekday trends shown as lines with standard deviation shaded: Limited (top), AC scenario (middle), MT-HVDC
scenario (bottom).

MT-HVDC scenario results for the combined Western and Eastern
Interconnections
Figure B-8 shows installed capacity post-nodal disaggregation for the complete MTHVDC scenario on the combined Western and Eastern Interconnection footprint. Figure
B-9 shows the resulting transmission expansion. The HVDC system is divided into
several section as described in Appendix A.10.

National Transmission Planning Study

115

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure B-8. Nodal disaggregated installed capacity for Western Interconnection and Eastern
Interconnection
Total installed capacity is shown in (a); (b) shows installed capacity by subregion.

Figure B-9. Transmission expansion for MT-HVDC scenario for Western and Eastern Interconnection

The generation dispatch for the complete MT-HVDC scenario is shown in Figure B-10
aggregated over the combined Western and Eastern Interconnection footprint for the
whole year and quarterly as well as split up by subregion. Except for the central section
of MISO, the regional breakdown highlights wind generation is using the MT-HVDC
network to get to load. The central section of MISO lies in the path of many HVDC
connections to the hub around Chicago, Illinois. Some of the converter stations around
this hub are placed in the MISO footprint and the power continues to Chicago on the AC
system. This helps explain the opposite direction of DC and AC imports in the bottom
of Figure B-10 for MISO Central.

National Transmission Planning Study

116

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure B-10. Generation dispatch for combined Western Interconnection and Eastern Interconnection
Panel (a) shows totals, (b) shows by quarter, and (c) shows by region.

Curtailment in the complete MT-HVDC scenario is shown in Figure B-11. The
distribution in terms of resource type skews much more heavily toward wind compared
to the Western Interconnection alone and is concentrated in the central regions of the
country (SPP and MISO).

National Transmission Planning Study

117

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure B-11. Curtailment for combined Western Interconnection and Eastern Interconnection footprint
Panel (a) shows aggregate curtailment, (b) shows aggregate curtailment by subregion, (c) shows aggregate curtailment by quarter,
and (d) shows average weekday trends as lines with ±1 standard deviation shaded.

The use of all HVDC corridors in both the Western Interconnection and Eastern
Interconnection are shown along with their geographic location in Figure B-12 and
Figure B-13. Except for the SPP north-south link and the FRCC to Southeast link, the
flows in the east are predominantly one-sided, showing a movement from the resourcerich western portion to the load areas in the east. This pattern contrasts with the use of
HVDC in the Western Interconnection, where flows—especially along the southern
border—are more bidirectional.

National Transmission Planning Study

118

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure B-12. Use of HVDC interregional interfaces in the MT-HVDC scenario (Western Interconnection)
Flow duration curves show the aggregate flow on the MT-HVDC lines between the subregions (including seam flows to the Eastern Interconnection).

National Transmission Planning Study

119

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure B-13. Use of HVDC interregional interfaces in the MT-HVDC scenario (Eastern Interconnection)
Flow duration curves show the aggregate flow on the MT-HVDC lines between the subregions. Note the HVDC links in PJM do not cross subregion boundaries and are therefore not
shown.

National Transmission Planning Study

120

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Appendix C. Tools
C.1 Sienna Modeling Framework
Production cost models simulate the least-cost optimal scheduling of electric generation
to meet system demand and transmission constraints. The functionality provided by the
Sienna modeling framework was used as the production-cost simulation engine in the
National Transmission Planning Study (NTP Study) (National Renewable Energy
Laboratory [NREL] 2024; Lara et al. 2021). The choice of the Sienna modeling
framework was a result of the range of libraries that make up Sienna being capable of
efficient and reproducible data management, programmatic access (which enables
reproducibility and dataset maintenance), and efficient storage and access to support
large-scale modeling, validation, and change management (all enabled through the
libraries part of Sienna\Data). Nodal production cost modeling in Sienna (via
Sienna\Ops) is transparent, programmatic, and scalable, which enables validation and
verification as well as the potential to develop further modeling functionality extensions.
Based in Julia (Bezanson et al. 2017), the Sienna framework enables speed of
development and execution combined with a wide range of available libraries while
leveraging the JuMP modeling language embedded in Julia for optimization (Lubin et al.
2023).

C.2 Grid Analysis and Visualization Interface
The Grid Analysis and Visualization Interface is a web application prototype for
visualizing large nodal bulk grid power simulations and is available at
http://github.com/NREL/GRAVI. It has been developed primarily as part of the NTP
Study with three main objectives:
• Aid in the dissemination of detailed nodal-level modeling results to stakeholders
• Visualize and interpret large-scale and data-intensive modeling results of the
contiguous U.S. bulk power system
• Aid in the transmission expansion planning that forms part of the translation of
zonal models into nodal models.
The tool provides a dynamic geospatial animated visualization of each timestep from a
simulation run along with complementary dynamic subplots. A screenshot of this for a
contiguous United States (CONUS)-wide nodal production cost model outcome is
shown in Figure C-1. It provides two controllable geospatial layers: one for bulk power
generation and the other for transmission use and flow.

National Transmission Planning Study

121

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure C-1. Screenshot-generated custom tool developed for the NTP Study (Grid Analysis and Visualization Interface)

National Transmission Planning Study

122

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Appendix D. Economic Methodology
D.1 Transmission Capital Cost Methodology
Transmission costs for the Limited Alternating Current (AC) Western Interconnection
case and the Interregional AC Western Interconnection case were developed based on
the transmission lines added to the GridView Production Cost Model by voltage class
(230 kilovolt [kV], 345 kV, and 500 kV). The Western Electricity Coordinating Council
(WECC) Transmission Calculator (Black & Veatch 2019) updated by E3 in 2019 (E3
2019) was the basis for calculating the capital costs for transmission. The WECC
Calculator multipliers for land ownership and terrain were used to estimate the cost of
added transmission. The WECC Environmental Viewer was used to calculate costs for
rights of way (ROWs), terrain, and land class. This appendix provides details on how
the ROW costs and terrain type multipliers were used to estimate the transmission
capital cost.

D.2 Developing the Right-of-Way Costs
Geospatial data for landcover, risk class, and Bureau of Land Management (BLM) zone
designation were acquired from the WECC Environmental Viewer (ICF, n.d.) (see
Figure D-1). The percent and total mileage of all transmission lines for each case
intersecting land cover types and BLM category were estimated using the ArcMap Pro
Tabulate Intersection Tool 68 (see Figure D-2 and Figure D-3). BLM land zones and their
costs are provided in Figure D-1. Land cover cost is estimated by the total mileage
multiplied by ROW width multiplied by BLM rental cost per acre (Table D-1). The ROW
width required by transmission lines by voltage class was acquired from Duke Energy
Transmission Guidelines (Table D-2) (Duke Energy, n.d.).

ArcGIS Pro. No date. “Tabulate Intersection Analysis.” Available at https://pro.arcgis.com/en/proapp/latest/tool-reference/analysis/tabulate-intersection.htm
68

National Transmission Planning Study

123

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure D-1. WECC environmental data viewer

Figure D-2. BLM zone classes and transmission lines for Interregional AC Western Interconnection case

National Transmission Planning Study

124

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Figure D-3. Costs associated with respective BLM zone

National Transmission Planning Study

125

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Table D-1. BLM Cost per Acre by Zone Number
BLM Cost Zone
Number

$/Acre

1

$83

2

$161

3

$314

4

$474

5

$653

6

$942

7

$1,318

8

$838

9

$4,520

10

$13,882

11

$27,765

12

$69,412

13

$138,824

14

$208,235

15

$277,647

Table D-2. Required Width for Transmission Lines by Voltage Class
Voltage Class
(kV)

Min Required
Width (ft)

Max Required
Width

44–115

68

100

230

125

150

500–525

180

200

D.3 Estimating the Cost by Terrain Type
Cost by land cover and terrain type was estimated as a multiplier (Table D-3) times the
cost per mile by voltage class (Table D-4). Voltage classes included single-circuit and
double-circuit additions. The cost by mile assumes the conductor type is aluminum
conductor steel reinforced (ACSR).

National Transmission Planning Study

126

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios
Table D-3. Land Cover and Terrain Classification Categories With Multiplier
Terrain Type

Terrain Type Identifier

Multiplier

Forested

1

2.25

Scrubbed/Flat

2

1

Wetland

3

1.2

Farmland

4

1

Desert/Barren Land

5

1.05

Urban

6

1.59

Rolling Hills (2%–8% slope)

7

1.4

Mountain (>8% slope)

8

1.75

Table D-4. Cost per Mile by Voltage Class
Voltage Class

Cost per Mile

230-kV Single Circuit

$1,024,335

230-kV Double Circuit

$1,639,820

345-kV Single Circuit

$1,434,290

345-kV Double Circuit

$2,295,085

500-kV Single Circuit

$2,048,670

500-kV Double Circuit

$3,278,535

D.4 Summary of Selected Literature Estimating the Economic
Benefits of Transmission
Building interregional transmission in the United States requires cooperation across
multiple balancing authorities and independent system operators (ISOs). Estimating the
economic benefits of interregional transmission provides critical information that could
help enable this cooperation. This section describes some of the issues highlighted in
previous literature regarding estimation of the economic benefits of transmission.
Transmission investment has generally been lower than the socially optimum amount.
William Hogan (1992), J. Bushnell and Stoft (1996), and J. B. Bushnell and Stoft (1997)
have shown transmission investments that are profitable to private ownership are also
economically efficient from a system perspective. However, many investments that are
socially beneficial are not profitable for privately owned transmission (Doorman and
Frøystad 2013; Egerer, Kunz and von Hirschhausen, Christian 2012; Gerbaulet and
Weber 2018). The primary reason is transmission enables lower-cost generation,
providing benefits to the system (and end consumers) that transmission owners do not
receive and hence is not used in their decision making. Several studies have shown
bargaining among utilities, ISOs, and so on will often not lead to an economically
efficient amount of transmission and these issues are amplified when market distortions
such as market power and higher negotiation costs are present (Cramton 1991;
Anderlini and Felli 2006; Joskow and Tirole 2005).

National Transmission Planning Study

127

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

William Hogan (2018) has written the most detailed methodology for the estimation and
computation of transmission projects. His methodology draws on the vast literature on
the economics of trade. A transmission can be modeled similar to a policy change that
opens up trade between two nations/states/and so on that was previously closed. Trade
economics is often taught from the perspective of comparative advantage, where both
nations specialize in some product and specialization occurs with trade—benefiting both
nations. With transmission, there is only one good available for trade. The region with
lower-cost generation will be an exporter, and the region with higher-cost generation will
be the importer. This can be modeled with the standard economic framework where
generators are suppliers, utilities are the consumers, and the payments made for
transmission are modeled similar to ad valorem taxes. Using this framework, producer
benefits can be measured with producer surplus, consumer benefits can be measured
with consumer surplus, and the payments to transmission owners are considered
“transmission rents.” Transmission rents are considered similar to ad valorem taxes
where the revenue is a transfer to another entity—that is, transmission rents should not
be considered losses, but the “wedge” the rents place between supply and demand
does create deadweight loss.
Several studies have considered similar transmission expansion needs in the European
Union (Sanchis et al. 2015; Neuhoff, Boyd, and Glachant 2012; Kristiansen et al. 2018).
Olmos, Rivier, and Perez-Arriaga (2018) showed a potential North Sea transmission
project joining six northern European countries could have net benefits up to €25.3
billion.
Within the United States, there have been limited national studies (Pfeifenberger, n.d.;
Stenclik, Derek and Deyoe, Ryan 2022). Some studies estimate the benefits of
intraregional transmission including in Midcontinent System Operator (MISO), California
Independent System Operator (CAISO), and New York Independent System Operator
(NYISO) (Gramlich, Rob 2022; Chowdhury and Le 2009; Conlon, Waite, and Modi
2019). These studies focus on estimating the benefits at the system level; i.e., they do
not consider the benefits that accrue to consumers, producers, or transmission owners
separately. The system benefits are computed by taking the reduction in generation
costs within each region and adjusting by subtracting import payments and adding
export revenues. William Hogan (2018) and CAISO (2017) showed the “system” method
yields the same total regional benefit when the benefits to consumers, producers, and
transmission rents are combined. However, the methods are equivalent only under the
assumption demand is fixed at each location.

National Transmission Planning Study

128

Chapter 3: Transmission Portfolios and Operations for 2035 Scenarios

Publication
DOE/GO-102024-6190
DOE/GO-102024-6258
Number | Publication
| September
Publication
Month
Month
2024
and Year
and Year
DOE/GO-102024-6259
| October
2024

National Transmission Planning Study

129
