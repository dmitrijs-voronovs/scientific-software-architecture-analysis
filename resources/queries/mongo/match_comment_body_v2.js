db.getCollection("scverse.scanpy").aggregate([
    {
        $unwind: "$comments_data"
    },
    {
        $addFields: {
            "text": {
                $replaceAll: {
                    input: "$comments_data.body",
                    find: "\r\n",
                    replacement: " "
                }
            }
        }
    },
    {
        $addFields: {
            "text": {
                $replaceAll: {
                    input: "$text",
                    find: "\n",
                    replacement: ""
                }
            }
        }
    },
    {
        $addFields: {
            "text": {
                $trim: { input: "$text" }
            }
        }
    },
    {
        $addFields: {
            "text_match": {$regexFindAll: { input: "$text", regex: /(base|you)(?=\w+?(?=[\s\p{P}]))/i }},
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