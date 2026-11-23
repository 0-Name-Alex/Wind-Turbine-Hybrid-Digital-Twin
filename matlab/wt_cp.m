function c = wt_cp(lam, beta)
%WT_CP  Power coefficient Cp(lambda, beta) - Heier model (textbook, Heier 2014).
%#codegen
li = 1 ./ (1 ./ (lam + 0.08*beta) - 0.035 ./ (beta.^3 + 1));
c  = max(0.5176*(116 ./ li - 0.4*beta - 5) .* exp(-21 ./ li) + 0.0068*lam, 0);
end
