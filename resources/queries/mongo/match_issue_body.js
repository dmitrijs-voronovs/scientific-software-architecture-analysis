db.getCollection("scverse.scanpy").aggregate([
    {
        $addFields: {
            text: "$body",
            text_match: { $regexFindAll: { input: "$body", regex: /(fast)\w*/i } },
        }
    },
    {
        $match: { "text_match.match": { $exists: true } }
    },
    {
        $unwind: "$text_match"
    },
        {
        $project: {
            text: 1,
            text_match: 1
        }
    }
])
