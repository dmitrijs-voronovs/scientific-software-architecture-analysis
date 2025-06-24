db.getCollection("scverse.scanpy").aggregate([{
    $unwind: "$comments_data"
}, {
    $addFields: {
        text: "$comments_data.body"
    }
}, {
    $match: {
        $or: [{
            "comments_data.user": {$not: {"$regex": /bot\b/i}}
        }, {
            "comments_data.user": {$in: ["olgabot", "hugtalbot"]}
        }]
    }
}, {
    $match: {
        "comments_data.user": {"$regex": /(olga|hugtal)bot\b/i}
        // "comments_data.user": { $not: { "$regex": /(?<!olga|hugtal)bot\b/i} }
    }
}
])