module XGBoost
  class PackedBooster
    def initialize(cvfolds)
      @cvfolds = cvfolds
    end

    def update(iteration)
      @cvfolds.each do |fold|
        fold.update(iteration)
      end
    end

    def eval_set(iteration)
      @cvfolds.map { |f| f.eval_set(iteration) }
    end

    def best_iteration
      @cvfolds[0].bst.best_iteration
    end

    def best_iteration=(iteration)
      @cvfolds.each do |fold|
        fold.best_iteration = iteration
      end
    end

    def best_score
      @cvfolds[0].bst.best_score
    end

    def best_score=(score)
      @cvfolds.each do |fold|
        fold.best_score = score
      end
    end

    def num_boosted_rounds
      @cvfolds[0].num_boosted_rounds
    end
  end
end
