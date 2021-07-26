% %function npyarray = mat2np(data)
% load interictal1.mat
% interictal=[]
% for i=1:1:1000
% temp=interictal1(:,i:i+999)
% interictal=[interictal;temp];
% end
% 
% %end
% result = cat(3, A,B,C,D,E)
%  test=cat(3,interictal1(:,1:1000),interictal1(:,1001:2000));
% test1(1,:,:)=test(:,:,1);
%Y = permute(result,[3 1 2]);
% %writeNPY(a,'a.NPY')


% clc
% clear all
% load b
%%%%%%%%%%%%%% Reading Interictal data%%%%%%%%%%%%%%%%%%5
% for k=10:1:99
%    myfilename = sprintf('Dog_1_interictal_segment_00%d.mat', k);
%    mydata = importdata(myfilename);
%    d=mydata.data;
%    for i=1:400:(floor(size(d,2)/400)*400)
%    a=cat(3,a,d(:,i:i+399));
%    end
% end

%%%%%%%%%%% Reading Preictal data %%%%%%%%%%%%%%%%%%%%%%%%
% for k=10:1:20
%    myfilename = sprintf('Dog_1_preictal_segment_00%d.mat', k);
%    mydata = importdata(myfilename);
%    d=mydata.data;
%    for i=1:400:(floor(size(d,2)/400)*400)
%    b=cat(3,b,d(:,i:i+399));
%    end
% end




% for k=10:1:99
%    myfilename = sprintf('Dog_1_interictal_segment_00%d.mat', k);
%    mydata = importdata(myfilename);
%    d=mydata.data;
%    a=cat(3,a,d);
% end
% 
% for k=100:1:480
%    myfilename = sprintf('Dog_1_interictal_segment_0%d.mat', k);
%    mydata = importdata(myfilename);
%    d=mydata.data;
%    a=cat(3,a,d);
% end
% myfilename = sprintf('Dog_1_interictal_segment_000%d.mat', 1);
% mydata = importdata(myfilename);
% d=my.data;
%%%%%%% Random Shuffle the data and labels %%%%%%%%%%%
[r c] = size(labels);
shuffledRow = randperm(r);
Y1 = Y1(shuffledRow, :);
retmat=X1(shuffledRow,:,:);