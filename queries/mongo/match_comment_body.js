db.getCollection("scverse.scanpy").aggregate([
    {
        $unwind: "$comments_data"
    },
    {
        $addFields: {
            "text": "$comments_data.body",
            "text_match": {$regexFindAll: { input: "$comments_data.body", regex: /(fast)(?=\w+?(?=[\s\p{P}]))/i }},
        }
    },
    {
        $match: {
            "text_match.match": { $exists: true }
        }
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