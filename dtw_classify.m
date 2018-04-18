function dtw_classify(training_file, test_file)
file_test = fopen(test_file);
test_line = fgets(file_test);
j = 0;
x_test = [];
y_test = [];
object_id_test = 0;
class_test = 0;
test_labels = [];
test_data = {};

%reading data file 
while ischar(test_line)
    if(length(test_line) < 47)
        if(length(test_line)~=2)
            if(strcmp(test_line(1:9),'object ID'))
                object_id_test = str2num(test_line(12:end));
            end
            if(strcmp(test_line(1:11),'class label'))
                class_test = str2num(test_line(14:end));
            end
            data_test = str2num(test_line);
            if(~isempty(data_test))
                x_test(end+1) = data_test(1);
                y_test(end+1) = data_test(2);
            end
        end
    else
        j = j + 1;
        if(j ~= 1)
            test_labels(j-1,1) = object_id_test;
            test_labels(j-1,2) = class_test;
            test_data{j-1,1} = x_test;
            test_data{j-1,2} = y_test;
            x_test = [];
            y_test = [];
        end
    end
    test_line = fgets(file_test);
    if test_line == -1
        test_labels(j-1,1) = object_id_test;
        test_labels(j-1,2) = class_test;
        test_data{j-1,1} = x_test;
        test_data{j-1,2} = y_test;
        x_test = [];
        y_test = [];
    end
end
fclose(file_test);

%separating data as training sets and test sets
file_training = fopen(training_file);
training_line = fgets(file_training);
i = 0;
x_training = [];
y_training = [];
object_id_training = 0;
class_training = 0;
training_labels = [];
training_data = {};

%manipulating data to perform dtw
while ischar(training_line)
    if(length(training_line) < 45)
        if(length(training_line)~=2)
            if(strcmp(training_line(1:9),'object ID'))
                object_id_training = str2num(training_line(12:end));
            end
            if(strcmp(training_line(1:11),'class label'))
                class_training = str2num(training_line(14:end));
            end
            data_training = str2num(training_line);
            if(~isempty(data_training))
                x_training(end+1) = data_training(1);
                y_training(end+1) = data_training(2);
            end
        end
    else
        i = i + 1;
        if(i ~= 1)
            training_labels(i-1,1) = object_id_training;
            training_labels(i-1,2) = class_training;
            training_data{i-1,1} = x_training;
            training_data{i-1,2} = y_training;
            x_training = [];
            y_training = [];
        end
    end
    training_line = fgets(file_training);
    if training_line == -1
        training_labels(i-1,1) = object_id_training;
        training_labels(i-1,2) = class_training;
        training_data{i-1,1} = x_training;
        training_data{i-1,2} = y_training;
        x_training = [];
        y_training = [];
    end
end
fclose(file_training);

best_cost = zeros(size(training_labels,1),2);
final_matrix = zeros(size(test_labels,1),1);

for i = 1:size(test_labels,1)
    test = [];
    test(:,1) = test_data{i,1}';
    test(:,2) = test_data{i,2}';
    for j = 1:size(training_labels,1)
        training = [];
        training(:,1) = training_data{j,1}';
        training(:,2) = training_data{j,2}';
        cost = zeros(size(test,1),size(training,1));
        cost(1,1) = sqrt(sum((test(1,:)-training(1,:)).*(test(1,:)-training(1,:))));
        for k = 2:size(test,1)
            cost(k,1) = cost(k-1,1) + sqrt(sum((test(k,:)-training(1,:)).*(test(k,:)-training(1,:))));
        end
        for q = 2:size(training,1)
            cost(1,q) = cost(1,q-1) + sqrt(sum((test(1,:)-training(q,:)).*(test(1,:)-training(q,:))));
        end
        for l = 2:size(test,1)
            for m = 2:size(training,1)
                val = [cost(l-1,m),cost(l,m-1),cost(l-1,m-1)];
                cost(l,m) = min(val) + sqrt(sum((test(l,:)-training(m,:)).*(test(l,:)-training(m,:))));
            end
        end
        best_cost(j,1) = cost(size(test,1),size(training,1));
        best_cost(j,2) = training_labels(j,2);
    end
    sorted_distance = sortrows(best_cost,1);
    if(sorted_distance(1,2) == test_labels(i,2))
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f, distance = %.2f\n', test_labels(i,1), sorted_distance(1,2), test_labels(i,2), 1, sorted_distance(1,1));
        final_matrix(i,1) = 1;
    else
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f, distance = %.2f\n', test_labels(i,1), sorted_distance(1,2), test_labels(i,2), 0, sorted_distance(1,1));
        final_matrix(i,1) = 0;
    end
end
fprintf('classification accuracy=%6.4f\n', mean(final_matrix));
end