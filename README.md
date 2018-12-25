# UPM-BigData-Spark
Spark application that creates a machine learning model for a real-world problem, using real-world data: Predicting the arrival delay of commercial flights


# Variable descriptions
n. | Forbidden |  Name |	Description
-- |-- |  ------| -------------
1	|	| Year |	1987-2008
2	|	| Month |	1-12
3	|	| DayofMonth |	1-31
4	|	| DayOfWeek |	1 (Monday) - 7 (Sunday)
5	|	| DepTime |	actual departure time (local, hhmm)
6	|	| CRSDepTime |	scheduled departure time (local, hhmm)
7	| x	| ArrTime |	actual arrival time (local, hhmm)
8	|	| CRSArrTime |	scheduled arrival time (local, hhmm)
9	|	| UniqueCarrier |	unique carrier code
10	|	| FlightNum |	flight number
11	|	| TailNum |	plane tail number
12	| x	| ActualElapsedTime |	in minutes
13	|	| CRSElapsedTime |	in minutes
14	| x	| AirTime |	in minutes
15	|	| ArrDelay |	arrival delay, in minutes
16	|	| DepDelay |	departure delay, in minutes
17	|	| Origin |	origin IATA airport code
18	|	| Dest |	destination IATA airport code
19	|	| Distance |	in miles
20	| x	| TaxiIn |	taxi in time, in minutes
21	|	| TaxiOut |	taxi out time in minutes
22	|	| Cancelled |	was the flight cancelled?
23	|	| CancellationCode |	reason for cancellation (A = carrier, B = weather, C = NAS, D = security)
24	|	x | Diverted |	1 = yes, 0 = no
25	|	x | CarrierDelay |	in minutes
26	|	x | WeatherDelay |	in minutes
27	|	x | NASDelay |	in minutes
28	|	x | SecurityDelay |	in minutes
29	|	x | LateAircraftDelay |	in minutes