WITH
    rulings AS (
        SELECT
            uuid,
            GROUP_CONCAT(text, ' ') as text
        FROM
            cardRulings
        GROUP BY
            uuid
    ),
    ranked_cards AS (
        SELECT
            c.*,
            ROW_NUMBER() OVER (
                PARTITION BY c.name, COALESCE(c.side, 'NULL_SIDE')
                ORDER BY c.uuid
            ) AS rn
        FROM
            cards c
        INNER JOIN
            cardLegalities cl ON cl.uuid = c.uuid
        WHERE
            cl.modern = 'Legal'
    )
SELECT
    rc.uuid,
    rc.name,
    rc.asciiName,
    rc.faceConvertedManaCost,
    rc.manaCost,
    rc.type,
    rc.power,
    rc.toughness,
    rc.loyalty,
    rc.defense,
    rc.colorIdentity,
    rc.colorIndicator,
    rc.colors,
    rc.edhrecRank,
    rc.edhrecSaltiness,
    rc.keywords,
    rc.language,
    rc.layout,
    rc.leadershipSkills,
    rc.manaValue,
    rc.rarity,
    rc.side,
    rc.types,
    rc.subtypes,
    rc.supertypes,
    rc.subsets,
    rc.text,
    rc.cardParts
FROM
    ranked_cards rc
    LEFT JOIN rulings r ON r.uuid = rc.uuid
WHERE
    rc.rn = 1;
