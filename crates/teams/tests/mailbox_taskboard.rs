use std::sync::Arc;
use std::time::Duration;

use opai_agents_teams::{
    ClaimOutcome, InMemoryMailbox, InMemoryTaskBoard, Mailbox, MemberId, Message, MessageKind,
    NewTask, TaskBoard, TaskId, TaskStatus, TaskUpdate, TeamError,
};

fn id(name: &str) -> MemberId {
    MemberId::new(name)
}

fn msg(from: &str, to: &str, body: &str) -> Message {
    Message::new(id(from), id(to), MessageKind::Text, body)
}

fn new_task(title: &str, deps: &[&TaskId]) -> NewTask {
    NewTask {
        title: title.into(),
        description: format!("do {title}"),
        blocked_by: deps.iter().map(|d| (*d).clone()).collect(),
        created_by: MemberId::lead(),
    }
}

// ---- mailbox ----

#[tokio::test]
async fn recv_waits_for_a_message_without_losing_it() {
    let mailbox = Arc::new(InMemoryMailbox::new());
    mailbox.register(&id("a")).await;

    let receiver = {
        let mailbox = Arc::clone(&mailbox);
        tokio::spawn(async move { mailbox.recv(&id("a")).await })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;
    mailbox.send(msg("b", "a", "hello")).await.unwrap();

    let got = receiver.await.unwrap().expect("a message");
    assert_eq!(got.body, "hello");
    assert_eq!(got.from, id("b"));
}

#[tokio::test]
async fn messages_arrive_in_order_and_drain_empties_the_inbox() {
    let mailbox = InMemoryMailbox::new();
    mailbox.register(&id("a")).await;
    for n in 0..3 {
        mailbox.send(msg("b", "a", &format!("m{n}"))).await.unwrap();
    }
    assert_eq!(mailbox.pending(&id("a")).await, 3);
    let bodies: Vec<_> = mailbox
        .drain(&id("a"))
        .await
        .into_iter()
        .map(|m| m.body)
        .collect();
    assert_eq!(bodies, ["m0", "m1", "m2"]);
    assert_eq!(mailbox.pending(&id("a")).await, 0);
}

#[tokio::test]
async fn a_full_inbox_refuses_instead_of_growing() {
    let mailbox = InMemoryMailbox::with_capacity(2);
    mailbox.register(&id("a")).await;
    mailbox.send(msg("b", "a", "1")).await.unwrap();
    mailbox.send(msg("b", "a", "2")).await.unwrap();
    let err = mailbox.send(msg("b", "a", "3")).await.unwrap_err();
    assert_eq!(err, TeamError::MailboxFull(id("a")));
    assert_eq!(mailbox.pending(&id("a")).await, 2);
}

#[tokio::test]
async fn sending_to_an_unknown_or_closed_inbox_fails() {
    let mailbox = InMemoryMailbox::new();
    assert_eq!(
        mailbox.send(msg("b", "ghost", "x")).await.unwrap_err(),
        TeamError::UnknownMember(id("ghost"))
    );
    mailbox.register(&id("a")).await;
    mailbox.close(&id("a")).await;
    assert!(mailbox.send(msg("b", "a", "x")).await.is_err());
}

#[tokio::test]
async fn close_wakes_a_waiting_receiver_with_none() {
    let mailbox = Arc::new(InMemoryMailbox::new());
    mailbox.register(&id("a")).await;
    let receiver = {
        let mailbox = Arc::clone(&mailbox);
        tokio::spawn(async move { mailbox.recv(&id("a")).await })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;
    mailbox.close(&id("a")).await;
    assert!(receiver.await.unwrap().is_none());
}

#[tokio::test]
async fn a_closed_inbox_still_hands_out_what_it_holds() {
    let mailbox = InMemoryMailbox::new();
    mailbox.register(&id("a")).await;
    mailbox.send(msg("b", "a", "last")).await.unwrap();
    mailbox.close(&id("a")).await;
    assert_eq!(mailbox.recv(&id("a")).await.unwrap().body, "last");
    assert!(mailbox.recv(&id("a")).await.is_none());
}

// ---- task board ----

#[tokio::test]
async fn only_one_of_many_concurrent_claims_wins() {
    let board = Arc::new(InMemoryTaskBoard::new());
    let task = board.create(new_task("t", &[])).await.unwrap();

    let mut handles = Vec::new();
    for n in 0..32 {
        let board = Arc::clone(&board);
        let task_id = task.id.clone();
        handles.push(tokio::spawn(async move {
            board.claim(&task_id, &id(&format!("m{n}"))).await.unwrap()
        }));
    }
    let mut claimed = 0;
    for handle in handles {
        match handle.await.unwrap() {
            ClaimOutcome::Claimed { .. } => claimed += 1,
            ClaimOutcome::AlreadyTaken { owner } => assert!(owner.is_some()),
            other => panic!("unexpected outcome {other:?}"),
        }
    }
    assert_eq!(claimed, 1);
    let stored = board.get(&task.id).await.unwrap().unwrap();
    assert_eq!(stored.status, TaskStatus::InProgress);
    assert!(stored.owner.is_some());
}

#[tokio::test]
async fn a_task_is_blocked_until_its_dependencies_complete() {
    let board = InMemoryTaskBoard::new();
    let first = board.create(new_task("first", &[])).await.unwrap();
    let second = board
        .create(new_task("second", &[&first.id]))
        .await
        .unwrap();

    let blocked = board.claim(&second.id, &id("b")).await.unwrap();
    assert_eq!(
        blocked,
        ClaimOutcome::Blocked {
            waiting_on: vec![first.id.clone()]
        }
    );

    assert!(matches!(
        board.claim(&first.id, &id("a")).await.unwrap(),
        ClaimOutcome::Claimed { .. }
    ));
    // Still blocked while the dependency is only in progress.
    assert!(matches!(
        board.claim(&second.id, &id("b")).await.unwrap(),
        ClaimOutcome::Blocked { .. }
    ));

    board
        .update(
            &first.id,
            &id("a"),
            TaskUpdate {
                status: Some(TaskStatus::Completed),
                result: Some("done".into()),
            },
        )
        .await
        .unwrap();
    assert!(matches!(
        board.claim(&second.id, &id("b")).await.unwrap(),
        ClaimOutcome::Claimed { .. }
    ));
}

#[tokio::test]
async fn creating_a_task_that_depends_on_nothing_known_fails() {
    let board = InMemoryTaskBoard::new();
    let err = board
        .create(new_task("t", &[&TaskId::new("nope")]))
        .await
        .unwrap_err();
    assert_eq!(err, TeamError::UnknownTask(TaskId::new("nope")));
}

#[tokio::test]
async fn only_the_owner_updates_and_completed_is_final() {
    let board = InMemoryTaskBoard::new();
    let task = board.create(new_task("t", &[])).await.unwrap();
    board.claim(&task.id, &id("a")).await.unwrap();

    let done = TaskUpdate {
        status: Some(TaskStatus::Completed),
        result: Some("ok".into()),
    };
    assert!(matches!(
        board.update(&task.id, &id("b"), done.clone()).await,
        Err(TeamError::NotOwner { .. })
    ));
    let finished = board
        .update(&task.id, &id("a"), done.clone())
        .await
        .unwrap();
    assert_eq!(finished.status, TaskStatus::Completed);
    assert_eq!(finished.result.as_deref(), Some("ok"));
    assert!(matches!(
        board.update(&task.id, &id("a"), done).await,
        Err(TeamError::InvalidTransition(_))
    ));
    assert_eq!(
        board.claim(&task.id, &id("c")).await.unwrap(),
        ClaimOutcome::AlreadyCompleted
    );
}

#[tokio::test]
async fn giving_a_task_up_makes_it_claimable_again() {
    let board = InMemoryTaskBoard::new();
    let task = board.create(new_task("t", &[])).await.unwrap();
    board.claim(&task.id, &id("a")).await.unwrap();
    board
        .update(
            &task.id,
            &id("a"),
            TaskUpdate {
                status: Some(TaskStatus::Pending),
                result: None,
            },
        )
        .await
        .unwrap();
    assert!(matches!(
        board.claim(&task.id, &id("b")).await.unwrap(),
        ClaimOutcome::Claimed { .. }
    ));
}

#[tokio::test]
async fn releasing_a_members_tasks_returns_only_unfinished_ones() {
    let board = InMemoryTaskBoard::new();
    let open = board.create(new_task("open", &[])).await.unwrap();
    let done = board.create(new_task("done", &[])).await.unwrap();
    let other = board.create(new_task("other", &[])).await.unwrap();
    for task in [&open, &done] {
        board.claim(&task.id, &id("a")).await.unwrap();
    }
    board.claim(&other.id, &id("b")).await.unwrap();
    board
        .update(
            &done.id,
            &id("a"),
            TaskUpdate {
                status: Some(TaskStatus::Completed),
                result: None,
            },
        )
        .await
        .unwrap();

    let released = board.release_owned_by(&id("a")).await.unwrap();
    assert_eq!(released, vec![open.id.clone()]);
    let tasks = board.list().await.unwrap();
    assert_eq!(tasks[0].status, TaskStatus::Pending);
    assert_eq!(tasks[1].status, TaskStatus::Completed);
    assert_eq!(tasks[2].owner, Some(id("b")));
}
