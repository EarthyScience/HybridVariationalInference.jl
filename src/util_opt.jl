"""
create a function (state, l) -> false that prints iter and loss each moditer
"""
callback_loss = (moditer) -> let moditer = moditer
    function (state, l)
        if state.iter % moditer == 1
            println("$(state.iter), $l")
        end
        # if state.iter >= 892
        #     println("$(state.iter)")
        #     error("stacktrace")
        # end
        return false
    end
end

callback_loss_fstate = (moditer, fstate) -> let moditer = moditer, fstate = fstate
    function (state, l)
        if state.iter % moditer == 1
            res_state = fstate(state)
            println("$(state.iter), $l, $res_state")
        end
        return false
    end
end




