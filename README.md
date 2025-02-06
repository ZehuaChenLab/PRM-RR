# PRM-RR
In recent years, the algorithm based on review text has been widely used in recommendation system, 
which can help mitigate the effect of sparsity in rating data within recommender algorithms. 
Existing methods typically employ a uniform model for capturing user and item features, but they are limited to the shallow
feature level, and the user’s personalized preferences and deep features of the item have not been fully
explored, which may affect the relationship between the two representations learned by the model. The
deeper relationship between them will affect the prediction results. Consequently, we propose a personalized
recommendation method based on rating matrix and review text denoted PRM-RR, which is used to deeply
mine user preferences and item characteristics. In the process of processing the comment text, we employ
ALBERT to obtain vector representations for the words present in the review text firstly. Subsequently,
taking into account that significant words and reviews bear relevance not solely to the review text but
also to the user’s individualized preferences, the proposed personalized attention module synergizes the
user’s personalized preference information with the review text vector, thereby engendering an enriched
review-based user representation. The fusion of the user’s review representation and rating representation
is accomplished through the feature fusion module using cross-modal attention, yielding the final user
representation. Lastly, we employ a factorization machine to predict the user’s rating for the item, thereby
facilitating the recommendation process. Experimental results on three benchmark datasets show that our
method outperforms the baseline algorithm in all cases, demonstrating that our method effectively improves
the performance of recommendations.
