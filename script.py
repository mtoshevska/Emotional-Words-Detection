from train import train_sent_detect, train_star_detect
from test import evaluate_sent_detect, generate_sent_detect_att_weights, \
    evaluate_star_detect, generate_star_detect_att_weights
from eval import evaluate


if __name__ == '__main__':
    train_sent_detect(r1=50, r2=500, padding_size=45, embedding_size=300, embedding_source='wikipedia',
                      learning_rate=0.0001, batch_size=256, num_epochs=50)
    evaluate_sent_detect(r1=50, r2=500, padding_size=45, embedding_size=300, embedding_source='wikipedia',
                         learning_rate=0.0001, epoch=50)
    generate_sent_detect_att_weights(r1=50, r2=500, padding_size=45, embedding_size=300, embedding_source='wikipedia',
                                     learning_rate=0.0001, epoch=50)
    evaluate(50, 500, 'SentDetect', 'w', 50, 'nrc')
    evaluate(50, 500, 'SentDetect', 'w', 50, 'yelp')

    train_star_detect(r1=50, r2=500, padding_size=45, embedding_size=300, embedding_source='wikipedia',
                      learning_rate=0.0001, batch_size=256, num_epochs=50)
    evaluate_star_detect(r1=50, r2=500, padding_size=45, embedding_size=300, embedding_source='wikipedia',
                         learning_rate=0.0001, epoch=50)
    generate_star_detect_att_weights(r1=50, r2=500, padding_size=45, embedding_size=300, embedding_source='wikipedia',
                                     learning_rate=0.0001, epoch=50)
    evaluate(50, 500, 'StarDetect', 'w', 50, 'nrc')
    evaluate(50, 500, 'StarDetect', 'w', 50, 'yelp')


    train_sent_detect(r1=50, r2=500, padding_size=45, embedding_size=200, embedding_source='twitter',
                      learning_rate=0.0001, batch_size=256, num_epochs=50)
    evaluate_sent_detect(r1=50, r2=500, padding_size=45, embedding_size=200, embedding_source='twitter',
                         learning_rate=0.0001, epoch=50)
    generate_sent_detect_att_weights(r1=50, r2=500, padding_size=45, embedding_size=200, embedding_source='twitter',
                                     learning_rate=0.0001, epoch=50)
    evaluate(50, 500, 'SentDetect', 't', 50, 'nrc')
    evaluate(50, 500, 'SentDetect', 't', 50, 'yelp')

    train_star_detect(r1=50, r2=500, padding_size=45, embedding_size=200, embedding_source='twitter',
                      learning_rate=0.0001, batch_size=256, num_epochs=50)
    evaluate_star_detect(r1=50, r2=500, padding_size=45, embedding_size=200, embedding_source='twitter',
                         learning_rate=0.0001, epoch=50)
    generate_star_detect_att_weights(r1=50, r2=500, padding_size=45, embedding_size=200, embedding_source='twitter',
                                     learning_rate=0.0001, epoch=50)
    evaluate(50, 500, 'StarDetect', 't', 50, 'nrc')
    evaluate(50, 500, 'StarDetect', 't', 50, 'yelp')

    train_sent_detect(r1=10, r2=500, padding_size=42, embedding_size=300, embedding_source='wikipedia',
                      learning_rate=0.0001, batch_size=256, num_epochs=50)
    evaluate_sent_detect(r1=10, r2=500, padding_size=42, embedding_size=300, embedding_source='wikipedia',
                         learning_rate=0.0001, epoch=50)
    generate_sent_detect_att_weights(r1=10, r2=500, padding_size=42, embedding_size=300, embedding_source='wikipedia',
                                     learning_rate=0.0001, epoch=50)
    evaluate(10, 500, 'SentDetect', 'w', 50, 'nrc')
    evaluate(10, 500, 'SentDetect', 'w', 50, 'yelp')

    train_star_detect(r1=10, r2=500, padding_size=42, embedding_size=300, embedding_source='wikipedia',
                      learning_rate=0.0001, batch_size=256, num_epochs=50)
    evaluate_star_detect(r1=10, r2=500, padding_size=42, embedding_size=300, embedding_source='wikipedia',
                         learning_rate=0.0001, epoch=50)
    generate_star_detect_att_weights(r1=10, r2=500, padding_size=42, embedding_size=300, embedding_source='wikipedia',
                                     learning_rate=0.0001, epoch=50)
    evaluate(10, 500, 'StarDetect', 'w', 50, 'nrc')
    evaluate(10, 500, 'StarDetect', 'w', 50, 'yelp')

    train_sent_detect(r1=10, r2=500, padding_size=42, embedding_size=200, embedding_source='twitter',
                      learning_rate=0.0001, batch_size=256, num_epochs=50)
    evaluate_sent_detect(r1=10, r2=500, padding_size=42, embedding_size=200, embedding_source='twitter',
                         learning_rate=0.0001, epoch=50)
    generate_sent_detect_att_weights(r1=10, r2=500, padding_size=42, embedding_size=200, embedding_source='twitter',
                                     learning_rate=0.0001, epoch=50)
    evaluate(10, 500, 'SentDetect', 't', 50, 'nrc')
    evaluate(10, 500, 'SentDetect', 't', 50, 'yelp')

    train_star_detect(r1=10, r2=500, padding_size=42, embedding_size=200, embedding_source='twitter',
                      learning_rate=0.0001, batch_size=256, num_epochs=50)
    evaluate_star_detect(r1=10, r2=500, padding_size=42, embedding_size=200, embedding_source='twitter',
                         learning_rate=0.0001, epoch=50)
    generate_star_detect_att_weights(r1=10, r2=500, padding_size=42, embedding_size=200, embedding_source='twitter',
                                     learning_rate=0.0001, epoch=50)
    evaluate(10, 500, 'StarDetect', 't', 50, 'nrc')
    evaluate(10, 500, 'StarDetect', 't', 50, 'yelp')
