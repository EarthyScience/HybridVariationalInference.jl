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

callback_loss_fstate = (moditer, fstate) -> let iter = 1, moditer = moditer, fstate = fstate
    function (state, l)
        if iter % moditer == 1
            res_state = fstate(state)
            println("$iter, $l, $res_state")
        end
        iter = iter + 1
        return false
    end
end




