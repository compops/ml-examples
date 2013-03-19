function [t,z] = regNN(X,alpha,beta,par)

% Calculate the derived features
    for m=1:par.NhiddenNodes
        z(m)=par.activationFunc(alpha(m,:)*X');
    end

    z=[1 z];
    
    % Calculate the outputs and the transformed outputs
    for k=1:par.NoutputNodes
        t(k)=beta(:,k)'*z';        
    end
end