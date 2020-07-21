function i =  innerp(h1,h2)
% h1, h2 should be on the same tangent space of x (same U and V);
i = h1.M(:)'*h2.M(:) + h1.Up(:)'*h2.Up(:) + h1.Vp(:)'*h2.Vp(:);
end