SELECT DISTINCT a.person
FROM SIGHTINGS a
WHERE a.location = 'Alaska Flat';

SELECT DISTINCT a.person
FROM SIGHTINGS a
WHERE a.location = 'Moreland Mill' and EXISTS (
  SELECT a2.location
  FROM SIGHTINGS a2
  WHERE a2.location = 'Steve Spring' and a2.person = a.person
  and a2.name = a.name);

SELECT DISTINCT  a.genus + ' ' + a.species
FROM FLOWERS a JOIN SIGHTINGS s ON a.comname = s.name
WHERE (s.person = 'Michael' or s.person = 'Robert') and EXISTS(
  SELECT f.elev
  FROM FEATURES f
  WHERE f.location = s.location and f.elev > 8250
);

SELECT DISTINCT f.map
FROM sightings a JOIN features f on a.location = f.location
WHERE a.name = 'Alpine penstemon' and DATENAME(m, a.sighted) = 'August';


CREATE VIEW genus_count AS
SELECT f.genus, COUNT(f.species) AS NUM_SPECIES
FROM flowers f
GROUP BY f.genus;

SELECT DISTINCT f.genus
FROM genus_count f
WHERE f.NUM_SPECIES > 1;


CREATE VIEW sighted_count AS
SELECT s.name, COUNT(s.location) AS NUM_SIGHTED
FROM sightings s
GROUP BY s.name;

SELECT a.name
FROM sighted_count a
WHERE a.NUM_SIGHTED = (SELECT MAX (b.NUM_SIGHTED) FROM sighted_count b);


SELECT DISTINCT s.person
FROM sightings s
WHERE NOT EXISTS(
  SELECT s2.person
  FROM sightings s2
  WHERE s2.person = s.person and EXISTS(
    SELECT f.class
    FROM features f
    WHERE f.location = s2.location and f.class = 'Tower'
  )
);




/*
Question 10
 */
CREATE VIEW summits AS
SELECT f.location
FROM features f
WHERE f.map = 'Sawmill Mountain' and f.class = 'Summit'
AND NOT f.location = 'Cerro Noroeste';

SELECT DISTINCT a.person
FROM sightings a join summits s on a.location = s.location
WHERE NOT EXISTS(
  SELECT s2.location
  FROM summits s2
  WHERE NOT a.location = s2.location
);


-- CREATE VIEW lats_top AS
-- SELECT DISTINCT TOP((SELECT COUNT(latitude) FROM features) - 19) f.latitude
-- FROM  features f
-- ORDER BY f.latitude DESC;
--
-- CREATE VIEW lats_ranges AS
-- SELECT f.latitude AS UPPER_LAT, MIN (
--   SELECT TOP (20) f2.latitude
--   FROM features f2
--   WHERE f2.latitude < f.latitude) AS LOWER_LAT
-- FROM lats_top f;

/*CREATE VIEW lats_dist AS
SELECT DISTINCT f.latitude
FROM features f;*/

CREATE VIEW lats_top AS
SELECT ROW_NUMBER() OVER (ORDER BY f.latitude DESC) as upper_row,
  ROW_NUMBER() OVER (ORDER BY f.latitude DESC) + 19 as lower_row,
  f.latitude
FROM lats_dist f;

CREATE VIEW lats_ranges AS
SELECT a.latitude as upper_lat, b.latitude as lower_lat
FROM lats_top a, lats_top b
WHERE b.upper_row = a.lower_row;

CREATE VIEW lats_sightings AS
SELECT f.latitude, COUNT(s.name) AS num_sighted
FROM features f JOIN sightings s ON f.location = s.location
GROUP BY f.latitude;

CREATE VIEW lats_concat as
SELECT f.upper_lat, f.lower_lat, (
  SELECT SUM(f2.num_sighted)
  FROM lats_sightings f2
  WHERE f2.latitude <= f.upper_lat and f2.latitude >= f.lower_lat
) AS num_sighted
FROM lats_ranges f;

SELECT f.upper_lat, f.lower_lat, f.num_sighted
FROM lats_concat f
WHERE f.num_sighted = (SELECT MAX (f2.num_sighted) FROM lats_concat f2);
