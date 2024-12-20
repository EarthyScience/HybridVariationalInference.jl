"""
create a function (state, l) -> false that prints iter and loss each moditer
"""
callback_loss = (moditer) -> let iter = 1, moditer = moditer
    function (state, l)
        if iter % moditer == 1
            println("$iter, $l")
        end
        iter = iter + 1
        return false
    end
end
