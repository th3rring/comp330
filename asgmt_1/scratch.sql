SELECT DISTINCT a.person
FROM SIGHTINGS a
WHERE a.location = 'Alaska Flat';
GO

SELECT DISTINCT a.person
FROM SIGHTINGS a
WHERE a.location = 'Moreland Mill' and EXISTS (
  FROM SIGHTINGS a2
  WHERE a2.location = 'Steve Spring' and a2.person = a.person
);
GO


